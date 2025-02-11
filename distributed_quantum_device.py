import os
from functools import partialmethod
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import torch.distributed
import torch.distributed.tensor
import torchquantum as tq
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torchquantum.macro import C_DTYPE, F_DTYPE
from torchquantum.functional import func_name_dict, apply_unitary_bmm, apply_unitary_einsum


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
        bsz = 1
        self.n_wires = n_wires
        self.device_name = device_name + "_distributed"
        self.bsz = bsz
        self.device = device

        _state = torch.zeros(2**self.n_wires, dtype=C_DTYPE)
        _state[0] = 1 + 0j  # type: ignore
        _state = torch.reshape(_state, [2] * self.n_wires)

        # set up distributed
        self.world_sz = world_sz
        rank = os.environ['LOCAL_RANK']
        self.rank = rank
        torch.cuda.set_device(f'{device}:{rank}')
        torch.distributed.init_process_group(world_size=world_sz)
        self.device_mesh = init_device_mesh(device, (world_sz,))

        sh_ = (bsz, ) + (2, ) * self.n_wires
        self.sh = sh_
        split_ix = int(np.log2(world_sz)) + 1
        self.split_ix = split_ix
        s1 = int(np.prod(sh_[:split_ix]))
        s2 = int(np.prod((2,) + sh_[split_ix:]))
        self.s1 = s1
        self.s2 = s2
        self.lin = torch.nn.Linear(s2, s1, bias=False, device=self.device)
        self.lin.weight.data = torch.view_as_real(_state).reshape((s1, -1))
        self.lin = parallelize_module(self.lin, self.device_mesh, parallelize_plan=RowwiseParallel())

        self.record_op = record_op
        self.op_history = []

    @property
    def states(self):
        sh = (self.bsz, ) + self.sh[self.split_ix:] + (2, )
        intermed = torch.view_as_complex(self.lin.weight.data.to_local().reshape(sh))

        return intermed

    @states.setter
    def states(self, value):
        print(f'{self.rank} in setter {torch.view_as_real(value).shape} {self.lin.weight.data.to_local().shape}')
        self.lin.weight.data = DTensor.from_local(
            torch.view_as_real(value).reshape((-1, )), device_mesh=self.device_mesh
        )

    def __del__(self):
        torch.distributed.destroy_process_group()

# set all to einsum
for name_, func_ in func_name_dict.items():
    func_einsum = partialmethod(func_, comp_method="einsum")
    setattr(DistributedQuantumDevice, name_, func_einsum)
