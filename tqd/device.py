import os
from functools import partialmethod
from typing import Union

import numpy as np
import torch
import torch.distributed
import torch.distributed.tensor
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

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

        # set up distributed
        self.world_sz = world_sz
        local_rank = int(os.environ['LOCAL_RANK'])
        global_rank = int(os.environ['RANK'])
        self.local_rank = local_rank
        self.global_rank = global_rank
        torch.cuda.set_device(f'{device}:{local_rank}')
        torch.distributed.init_process_group(world_size=world_sz)
        self.device_mesh = init_device_mesh(device, (world_sz,))

        self.log2_devices = int(np.ceil(np.log2(world_sz)))
        # use 1st dim for batching, last dim for real/imag
        self.local_shape = (bsz, ) + (2, ) * (self.n_wires - self.log2_devices) + (1, ) * self.log2_devices + (2, )
        self._wire_order = list(range(self.n_wires))
        # shard along last wire dimensions: assume that first computations use lower number wires
        self.sharded_wires = [self.n_wires - 1 - i for i in range(self.log2_devices)]
        self._states = torch.zeros(self.local_shape)
        placements = [Shard(i+1) for i in self.sharded_wires]
        if self.global_rank == 0:
            self._states[(0, ) * self._states.ndim] = 1
        self._states = DTensor.from_local(self._states, self.device_mesh, placements)
    
    def canonicalize(self):
        self._states = self._states.permute((0, ) + tuple(1 + np.argsort(self._wire_order)) + (self.n_wires+1, ))
        self._wire_order = list(range(self.n_wires))
    
    @property
    def states(self):
        return self._states.permute((0, ) + tuple(1 + np.argsort(self._wire_order)) + (self.n_wires+1, ))

    def __del__(self):
        torch.distributed.destroy_process_group()

    def maybe_reshard(self, wires):
        """
        currently assumes 2Q gates with connectivity < n_wires/2
        """
        cur_sharded_qubits = {s_.dim-1 for s_ in self._states.placements}
        overlap = set(wires) & cur_sharded_qubits
        if overlap:  # only if wires affect sharded dimensions
            new_qubit_sharding = cur_sharded_qubits - overlap
            usable_qubits = sorted(set(range(self._states.ndim - 1)) - (set(wires) | cur_sharded_qubits))
            # hardcode: 2qubit gates only
            min_wire = min(wires)
            max_wire = max(wires)
            # hardcode: n_wires > 2 * connectivity
            if max_wire - min_wire > min_wire + self.n_wires - max_wire:
                min_wire, max_wire = max_wire, min_wire + self.n_wires
            best_usable_qubits = [q_ for q_ in usable_qubits if q_ < min_wire]
            for i in range(len(overlap)):
                new_qubit_sharding.add(best_usable_qubits[-1-i])
            # all2all; add 1 for the batch dimension!
            new_dim_sharding = [i + 1 for i, w in enumerate(self._wire_order) if w in new_qubit_sharding]
            self._states = self._states.redistribute(self.device_mesh, placements=[Shard(d) for d in new_dim_sharding])


# Give DQD methods, so we can write e.g. `qdev.ry(wires=[0])`
for name_ in matrices.GATE_MAT_DICT.keys():
    func = partialmethod(getattr(functional, name_))
    setattr(DistributedQuantumDevice, name_, func)
