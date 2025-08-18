import os
from functools import partialmethod
from typing import Union

import numpy as np
import torch
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from . import functional, matrices


class DistributedQuantumDevice:
    def __init__(
        self,
        n_wires: int,
        bsz: int = 1,
        device_name: str = "default",
        device: Union[torch.device, str] = "cuda",
        record_op: bool = False,
        world_sz: int = 1,
        shared_seed: int = 20740,
        invertible: bool = False,
        max_dtensor_dims: int = 16
    ):
        """A quantum device that contains the quantum state vector.
        Args:
            n_wires: number of qubits
            device_name: name of the quantum device
            bsz: batch size of the quantum state
            device: which classical computing device to use, 'cpu' or 'cuda'
            record_op: whether to record the operations on the quantum device and then
                they can be used to construct a static computation graph
        """
        # number of qubits
        # the states are represented in a multi-dimension tensor
        # from left to right: qubit 0 to n
        self.n_wires = n_wires
        self.device_name = device_name + "_distributed"
        self.bsz = bsz
        self.device = device
        self.shared_seed = shared_seed

        self.record_op = record_op
        self.op_history = []
        self.invertible = invertible

        # set up distributed
        self.world_sz = world_sz
        local_rank = int(os.environ['LOCAL_RANK'])
        global_rank = int(os.environ['RANK'])
        self.local_rank = local_rank
        self.global_rank = global_rank
        if device =='cuda':
            torch.cuda.set_device(f'cuda:{local_rank}')
        self.device_mesh = None
        self.log2_devices = int(np.ceil(np.log2(world_sz)))
        if world_sz > 1:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl", init_method='env://', rank=global_rank, world_size=world_sz)
            self.device_mesh = init_device_mesh(device, (2,) * self.log2_devices)

        # First row of self._groupings indicates group number, second row indicates relative position
        self._groupings = torch.zeros((2, self.n_wires), dtype=torch.int)
        # use 1st dim for batching, last dim for real/imag
        if self.n_wires < max_dtensor_dims - 2:
            self.local_shape = (bsz, ) + (2, ) * (self.n_wires - self.log2_devices) + (1, ) * self.log2_devices + (2, )
            self._groupings[0] = torch.arange(self.n_wires) # Each qubit is own group
        else:
            # Number of possible dimensions for grouping equals total dimension number minus batching dim, real/imag dim, sharding dims, and two single-qubit dims
            num_grouped_dims = max_dtensor_dims - 4 - self.log2_devices
            num_grouped_qubits = self.n_wires - self.log2_devices - 2

            group_nums = [num_grouped_qubits//num_grouped_dims + int(i < num_grouped_qubits%num_grouped_dims) for i in range(num_grouped_dims)]
            self.local_shape = (bsz, ) + (2, ) + tuple([2**qubits for qubits in group_nums]) + (2, ) + (1, ) * self.log2_devices + (2, )

            # Arrange groupings according to grouped_dimensions
            # First and last groupings always contain only 1 qubit
            # Second row designation of -1 includes ungrouped qubit, -2 designations sharded qubit
            self._groupings[:,0] = torch.Tensor([1,-1], dtype=torch.int)
            qubit_idx = 1
            for i in in range(len(group_nums)):
                self._groupings[0,qubit_idx:qubit_idx + group_nums[i]] = i + 2
                if group_nums[i] == 1:
                    self_groupings[1, qubit_idx] = -1
                else:
                    self._groupings[1,qubit_idx:qubit_idx + group_nums[i]] = torch.arange(group_nums[i])
                qubit_idx += group_nums[i]
            # qubit_idx now indicates end of grouped qubits
            self._groupings[0,qubit_idx] = len(group_nums) + 2
            self._groupings[1,qubit_idx] = -1
            self._groupings[0,qubit_idx + 1:] = len(group_nums) + 3 + torch.arange(self.log2_devices)
            self._groupings[1,qubit_idx + 1:] = -2
        
        self.last_unsharded = qubit_idx # last unsharded group always contains one qubit
        self.num_dims = max_dtensor_dims
        self.reset_states()

    def reset_states(self):
        # shard along last wire dimensions: assume that first computations use lower number wires
        sharded_wires = (self._groupings[:,(self._groupings[1] == -1)][0][-self.log2_devices:]).tolist()
        self._states = torch.zeros(self.local_shape, device=self.device)
        if self.global_rank == 0:
            self._states[(slice(None), ) + (0, ) * (self._states.ndim - 1)] = 1
        if self.world_sz > 1:
            placements = [Shard(i+1) for i in sharded_wires]
            self._states = DTensor.from_local(self._states, self.device_mesh, placements)

        if self.invertible:
            self._invertible_dummy = torch.tensor(0, dtype=self._states.dtype, device=self._states.device)
            if self.world_sz > 1:
                self._invertible_dummy = DTensor.from_local(
                    self._invertible_dummy.expand_as(self._states.to_local()),
                    self.device_mesh, placements
                )
            else:
                self._invertible_dummy = self._invertible_dummy.expand_as(self._states)
        else:
            self._invertible_dummy = None

    def interchange_qubits(self, state: Union[DTensor, torch.Tensor], grouping: torch.Tensor, wire1: int, wire2: int) -> new_state: Union[DTensor, torch.Tensor], new_grouping: torch.Tensor:
        '''
        Interchanges two qubits within a given statevector. Does not reshard on its own and can only interchange sharded qubits with ungrouped qubits.
        Args:
            state: Tensor or DTensor containing state information
            grouping: grouping information for state
            wire1: first qubit index
            wire2: second qubit index
        Returns:
            new_state: Tensor or DTensor with interchanged qubits
            new_grouping: new grouping information for new_state
        '''
        wire_info = grouping[:,[wire1, wire2]]
        if wire1 == wire2:
            return state, grouping
        elif (wire_info[1] > -1).any():
            if (wire_info[1] == -2).any(): # Cannot interchange sharded qubit with grouped unsharded qubit
                return state, grouping
            elif (wire_info[1] == -1).any(): # Exactly one wire is ungrouped
                if wire_info[1,0] == -1: # Presume from now on that wire2 is the ungrouped wire
                    return self.interchange_qubits(state, grouping, wire2, wire1)
                else:
                    lone_wire_idx = wire_info[0,1]
                    # Get index and size of wire1 group along with relative position of wire1
                    grouped_wire_info = [wire_info[0,0].item(), wire_info[1,0].item(), grouping[0, grouping[0]==wire_info[0,0]].sum().item()]
                    # grouped wires cannot be leftmost or rightmost dimensions; get index and size of left and right groups
                    left_group_info = [grouped_wire_info[0] - 1, grouping[0, grouping[0]==(grouped_wire_info[0] - 1)].sum().item()]
                    right_group_info = [grouped_wire_info[0] + 1, grouping[0, grouping[0]==(grouped_wire_info[0] + 1)].sum().item()]
 
                    need_left_interchange = (lone_wire_idx == left_group_info[0]) and (grouped_wire_info[1] > 0)
                    need_right_interchange = (lone_wire_idx == right_group_info[0]) and (grouped_wire_info[1] < grouped_wire_info[2] - 1)
                    need_interchange = need_left_interchange or need_right_interchange
                    if need_interchange:
                        all_ungrouped_idxs = grouping[0,[grouping[1] == -1]]
                        # Pick a new ungrouped qubit that isn't adjacent to wire1
                        helper_idx = all_ungrouped_idxs[((all_ungrouped_idxs != left_group_info[0])&(all_ungrouped_idxs != right_group_info[0]))][0]
                        helper_qubit = (torch.nonzero(grouping[0] == helper_idx).flatten()[0]).int()
                        new_state, new_grouping = self.interchange_qubits(state, grouping, wire1, helper_qubit)
                        new_state, new_grouping = self.interchange_qubits(new_state, new_grouping, wire1, wire2)
                        new_state, new_grouping = self.interchange_qubits(new_state, new_grouping, helper_qubit, wire2)
                        return new_state, new_grouping
                    else:
                        # Get temporary tensor shape
                        state_shape = list(state.shape)
                        temp_shape = state_shape
                        temp_shape[grouped_wire_info[0]] = 2
                        temp_shape[left_group_info[0]] = 2**(left_group_info[1] + grouped_wire_info[1])
                        temp_shape[right_group_info[0]] = 2**(right_group_info[1] + grouped_wire_info[2] - 1 - grouped_wire_info[1])
                        
                        # Reshape to isolate grouped wire dimension, interchange wires, and then reshape back
                        new_state = state.reshape(temp_shape)
                        new_state,_ = self.interchange_dims(new_state, grouping, grouped_wire_info[0], lone_wire_idx, new_state.ndim)
                        new_state = new_state.reshape(state_shape)
                        
                        new_grouping = grouping.detach().clone()
                        new_grouping[:,wire1], new_grouping[:,wire2] = new_grouping[:,wire2], new_grouping[:,wire1]
                        return new_state, new_grouping
            else: # If both wires are grouped, use ungrouped wire as medium of interchange
                helper_qubit = (torch.nonzero(grouping[1] == -1).flatten()[0]).int()
                new_state, new_grouping = self.interchange_qubits(state, grouping, wire2, helper_qubit)
                new_state, new_grouping = self.interchange_qubits(new_state, new_grouping, wire1, wire2)
                new_state, new_grouping = self.interchange_qubits(new_state, new_grouping, helper_qubit, wire1)
                return new_state, new_grouping
        else:
            # Between ungrouped and sharded qubits, interchanging qubits is the same as interchanging dimensions
            return self.interchange_dims(state, grouping, wire_info[0,0], wire_info[0,1], state.ndim)

    def interchange_dims(self, state: Union[DTensor, torch.Tensor], grouping: torch.Tensor, dim1: int, dim2: int, num_dims: int) -> new_state: Union[DTensor, torch.Tensor], new_grouping = torch.Tensor:
        '''
        Interchanges two tensor dimensions, either groups of qubits or ungrouped qubits, and updates grouping tensor
        Note that relative orders within dimensions (or designations of ungrouped or sharded qubits, remain unchanged
        Args:
            state: Tensor or DTensor containing state info. Must containg self.num_dims dims
            grouping: Tensor containing grouping info for state
            dim1: first dimension
            dim2: second dimension
            num_dims: number of dimensions in states
        Results:
            new_state: new state with dimensions interchanged
            new_grouping: new grouping with dimensions interchanged
        '''
        permute_list = list(range(num_dims))
        permute_list[dim1], permute_list[dim2] = dim2, dim1
        new_state = state.permute(permute_list)
        new_grouping = grouping.detach().clone()
        new_grouping[0,new_grouping[0]==dim1], new_grouping[0, new_grouping[0]==dim2] = dim2, dim1
        return new_state, new_grouping

    def noncanonical_states(self) -> Union[DTensor, torch.Tensor], torch.Tensor:
        '''
        Obtain device states and groupings without canonicalization (used for measurements)
        The groupings are detached and cloned to prevent interference but the states are not to avoid unnecessary computations.
        The expectation is that someone calling this function from outside the device would modify the statevector in such a way as to result in a copy of it,
        instead of modifying inplace.
        '''
        return self._states, self._groupings.detach().clone()

    def canonicalize(self):
        # self._states, self._groupings, self._invertible_dummy = self.reorder(self._states, self._groupings, self._invertible_dummy)
        # First maybe_reshard final self.log2_devices qubits
        sharded_wires = torch.nonzero(self._groupings[1] == -2).flatten()
        wires_to_unshard = sharded_wires[sharded_wires < (self.n_wires - self.log2_devices)].tolist()
        do_not_shard = list(range(self.n_wires - self.log2_devices))
        for wire in wires_to_unshard:
            self.maybe_reshard(wires=[wire], ignore=do_not_shard)
        # Final qubits are all sharded (though possibly unordered); now obtain wire ordering
        wire_order = torch.nonzero(self._groupings[0] == 1).flatten() # First wire group is always ungrouped
        for i in range(2, self.num_dims - 1):
            group_mask = self._groupings[0] == i
            group_wires = torch.nonzero(group_mask).flatten()
            if len(group_wires) > 1:
                relative_order = self._groupings[1,group_mask].argsort()
                group_wires = group_wires[relative_order]
            wire_order = torch.cat((wire_order, group_wires))

        # Now interchange qubits until they are all in the proper order
        for i in range(self.n_wires):
            if wire_order[i] == i:
                continue
            else:
                if self._invertible_dummy is not None:
                    self._invertible_dummy,_ = self.interchange_qubits(self._invertible_dummy, self._groupings, i, wire_order[i])
                self._states, self._groupings = self.interchange_qubits(self._states, self._groupings, i, wire_order[i])
                wire_order[wire_order == i], wire_order[i] = wire_order[i], i
    
    @property
    def states(self):
        self.canonicalize()
        return self._states.detach().clone()

    @property
    def invertible_dummy(self):
        if self._invertible_dummy is not None:
            self.canonicalize()
            return self._invertible_dummy.detach().clone()

    def maybe_reshard(self, wires: list[int], ignore: list[int], inverse: bool=False):
        """
        If the current sharding splits the statevector in the dimension that is acted upon by the
        gate, picks a new dimension to shard over and redistributes the statevector accordingly.
        The new sharding dimension is picked assuming a ladder ansatz where 2Q gates are applied
        between wires (i, i+c) looping through i increasing and where c is connectivity, and any
        1Q gates are applied to the i+c wire (example below).
        Incurs an all2all.

        If the sharded dimension is not acted upon by the gate, does nothing.

        For example, in a circuit with 6 qubits, suppose qubit wire 4 is sharded. We assume
        c < 6/2 = 3, so max(c) == 2. We will need to reshard when the 2Q gate operates on
        wires (2, 4). Knowing that the next 2Q gates will operate on either (3, 5) or (4, 5)
        and any intermediate 1Q gates would operate on wire 5, we would prefer to reshard wire 2
        to avoid resharding for as long as possible.

        As future work, we could make this a little stronger by examining the current connectivity.

        Args
            wires: (`list` of `int`) indices of qubits that will be acted upon by the gate. Presumed to contain at most two qubits
            ignore: indices of qubits that will not be sharded
            inverse: when going in reverse direction for invertible backpropagation, we need to
                invert the dimension picking logic since the reversed ladder decreases in indices.

        Returns:
            None
        """
        if self.world_sz <= 1:
            return
        cur_sharded_qubits = set(torch.nonzero(self._groupings[1] == -2).flatten().tolist())
        overlap = set(wires) & cur_sharded_qubits
        if overlap:  # only if wires affect sharded dimensions
            new_qubit_sharding = cur_sharded_qubits - overlap
            usable_qubits = sorted(set(range(self.n_wires)) - (set(wires) | cur_sharded_qubits | set(ignore)))
            # hardcode: 2qubit gates only
            min_wire = min(wires)
            max_wire = max(wires)
            # hardcode: n_wires > 2 * connectivity
            if max_wire - min_wire > min_wire + self.n_wires - max_wire:
                if inverse:
                    min_wire, max_wire = max_wire - self.n_wires, min_wire
                else:
                    min_wire, max_wire = max_wire, min_wire + self.n_wires
            # this happens to be the same for inverse and not!
            best_usable_qubits = [q_ for q_ in usable_qubits if q_ > max_wire] + [q_ for q_ in usable_qubits if q_ < min_wire]
            uncoupled_qubits = torch.nonzero(self._groupings[1] == -1).flatten().tolist()
            switch_qubits = []
            for i in range(len(overlap)):
                if inverse:
                    next_usable_qubit = best_usable_qubits[i]
                else:
                    next_usable_qubit = best_usable_qubits[-1-i]
                # Move selected usable qubits into uncoupled dimensions (there are always at least 2 uncoupled qubits and at most 2 qubits selected)
                if self._invertible_dummy is not None:
                    self._invertible_dummy,_ = self.interchange_qubits(self._invertible_dummy, self._groupings, next_usable_qubit, uncoupled_qubits[i])
                self._states, self._groupings = self.interchange_qubits(self._states, self._groupings, next_usable_qubit, uncoupled_qubits[i])
                switch_qubits.append(next_usable_qubit)
            for i in range(len(switch_qubits)):
                # Interchange uncoupled selected qubits and sharded qubits in anticipation of resharding
                if self._invertible_dummy is not None:
                    self.invertible_dummy,_ = self.interchange_qubits(self._invertible_dummy, self.groupings, switch_qubits[i], list(overlap)[i])
                self._states, self._groupings = self.interchange_qubits(self._states, self.groupings, switch_qubits[i], list(overlap)[i])
                self._groupings[1,switch_qubits[i]], self._groupings[1,list(overlap)[i]] = -2, -1
            new_qubit_sharding = new_qubit_sharding.union(switch_qubits)
            # all2all
            new_dim_sharding = [self._groupings[0, w] for w in new_qubit_sharding].sort()
            self._states = self._states.redistribute(self.device_mesh, placements=[Shard(d) for d in new_dim_sharding])
            if self._invertible_dummy is not None:
                self._invertible_dummy = self._invertible_dummy.redistribute(self.device_mesh, placements=[Shard(d) for d in new_dim_sharding])


# Give DQD methods, so we can write e.g. `qdev.ry(wires=[0])`
for name_ in matrices.GATE_MAT_DICT.keys():
    func = partialmethod(getattr(functional, name_))
    setattr(DistributedQuantumDevice, name_, func)
