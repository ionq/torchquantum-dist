import os
from functools import partialmethod
from typing import Union

import numpy as np
import torch
import torch.distributed
import torch.distributed.tensor
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from . import functional


class DistributedQuantumDevice:
    def __init__(
        self,
        n_wires: int,
        device_name: str = "default",
        device: Union[torch.device, str] = "cuda",
        record_op: bool = False,
        world_sz: int = 1,
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
        bsz = 2  # batch ix 0 is real, batch ix 1 is imag
        self.n_wires = n_wires
        self.device_name = device_name + "_distributed"
        self.bsz = bsz
        self.device = device

        self.record_op = record_op
        self.op_history = []

        # set up distributed
        self.world_sz = world_sz
        rank = os.environ['LOCAL_RANK']
        self.rank = rank
        torch.cuda.set_device(f'{device}:{rank}')
        torch.distributed.init_process_group(world_size=world_sz)
        self.device_mesh = init_device_mesh(device, (world_sz,))

        # shard along last dimensions: assume that first computations use lower number wires
        self.log2_devices = int(np.ceil(np.log2(world_sz)))
        self.local_shape = (2, ) + (2, ) * (self.n_wires - self.log2_devices) + (1, ) * self.log2_devices
        self.full_shape = (2, ) + (2, ) * self.n_wires
        _states = torch.zeros(self.local_shape)
        placements = [Shard(self.n_wires-i) for i in range(self.log2_devices)]
        if self.rank == '0':
            _states[(0,) * _states.ndim] = 1
        self.states = DTensor.from_local(_states, self.device_mesh, placements)

    def __del__(self):
        torch.distributed.destroy_process_group()

    def maybe_reshard(self, wires):
        """
        currently assumes 2Q gates with connectivity < n_wires/2
        """
        cur_sharded_qubits = {s_.dim-1 for s_ in self.states.placements}
        overlap = set(wires) & cur_sharded_qubits
        if overlap:  # only if wires affect sharded dimensions
            new_qubit_sharding = cur_sharded_qubits - overlap
            usable_qubits = sorted(set(range(self.states.ndim - 1)) - (set(wires) | cur_sharded_qubits))
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
            self.states = self.states.redistribute(self.device_mesh, placements=[Shard(i+1) for i in new_qubit_sharding])


# Give DQD methods, so we can write e.g. `qdev.ry(wires=[0])`
for name_ in functional.FUNC_NAMES:
    func_einsum = partialmethod(getattr(functional, name_), comp_method="bmm")
    setattr(DistributedQuantumDevice, name_, func_einsum)
