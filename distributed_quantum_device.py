import functools
import importlib
import itertools
import os
from functools import partial, partialmethod
from typing import Callable, Union

import numpy as np
import torch
import torch.distributed
import torch.distributed.tensor
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torchquantum.macro import ABC, ABC_ARRAY, F_DTYPE

PAULIS = {}
RS = {}
for letter in ['x', 'y', 'z']:
    PAULIS[letter] = importlib.import_module(f'.pauli{letter}', f'torchquantum.functional')
    RS[letter] = importlib.import_module(f'.r{letter}', f'torchquantum.functional')

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

        self.log2_devices = int(np.ceil(np.log2(world_sz)))
        self.local_shape = (2, ) + (1, ) * self.log2_devices + (2, ) * (self.n_wires - self.log2_devices)
        self.full_shape = (2, ) + (2, ) * self.n_wires
        _states = torch.zeros(self.local_shape)
        self.placements = [Shard(i+1) for i in range(self.log2_devices)]
        if self.rank == '0':
            _states[(0,) * _states.ndim] = 1
        self.states = DTensor.from_local(_states, self.device_mesh, self.placements)

    def __del__(self):
        torch.distributed.destroy_process_group()

def apply_unitary_einsum(state, mat, wires):
    """Apply the unitary to the statevector using torch.einsum method.

    Args:
        state (torch.Tensor): The statevector as a local DTensor
        mat (torch.Tensor): The unitary matrix of the operation.
        wires (int or List[int]): Which qubit the operation is applied to.

    Returns:
        torch.Tensor: The new statevector.

    """
    device_wires = wires

    # minus one because of batch
    total_wires = len(state.shape) - 1

    if len(mat.shape) > 2:
        is_batch_unitary = True
        bsz = mat.shape[0]
        shape_extension = [bsz]
    else:
        is_batch_unitary = False
        shape_extension = []

    mat = mat.view(shape_extension + [2] * len(device_wires) * 2)

    mat = distribute_tensor(mat, state.device_mesh, [Replicate()])

    # Tensor indices of the quantum state
    state_indices = ABC[:total_wires]

    # Indices of the quantum state affected by this operation
    affected_indices = "".join(ABC_ARRAY[list(device_wires)].tolist())

    # All affected indices will be summed over, so we need the same number
    # of new indices
    new_indices = ABC[total_wires: total_wires + len(device_wires)]

    # The new indices of the state are given by the old ones with the
    # affected indices replaced by the new_indices
    new_state_indices = functools.reduce(
        lambda old_string, idx_pair: old_string.replace(idx_pair[0], idx_pair[1]),
        zip(affected_indices, new_indices),
        state_indices,
    )

    state_indices = ABC[-1] + state_indices
    new_state_indices = ABC[-1] + new_state_indices
    if is_batch_unitary:
        new_indices = ABC[-1] + new_indices

    # We now put together the indices in the notation numpy einsum
    # requires
    einsum_indices = (
        f"{new_indices}{affected_indices}," f"{state_indices}->{new_state_indices}"
    )

    new_state = torch.einsum(einsum_indices, mat, state)

    return new_state

def gate_wrapper(
    name, mat, method, q_device,
    wires, params=None        
):
    if params is not None:
        if not isinstance(params, torch.Tensor):
            # this is for directly inputting parameters as a number
            params = torch.tensor(params, dtype=F_DTYPE)

        if params.dim() == 1:
            params = params.unsqueeze(-1)
        elif params.dim() == 0:
            params = params.unsqueeze(-1).unsqueeze(-1)
    wires = [wires] if isinstance(wires, int) else wires

    if q_device.record_op:
        q_device.op_history.append(
            {
                "name": name,  # type: ignore
                "wires": np.array(wires).squeeze().tolist(),
                "params": params.squeeze().detach().cpu().numpy().tolist()
                if params is not None
                else None,
                "trainable": params.requires_grad if params is not None else False,
            }
        )

    # in dynamic mode, the function is computed instantly
    if isinstance(mat, Callable):
        matrix = mat(params)
    else:
        matrix = mat

    assert np.log2(matrix.shape[-1]) == len(wires)

    matrix_real, matrix_imag = torch.view_as_real(matrix).split(1, dim=-1)
    matrix_real, matrix_imag = matrix_real[..., 0], matrix_imag[..., 0]
    if q_device.device_name=="noisedevice":
        raise ValueError("In `gate_wrapper`: `noisedevice` not supported yet")
        density = q_device.densities
        print(density.shape)
        if method == "einsum":
            raise ValueError("In `gate_wrapper`: `einsum` not supported for `method`")
        elif method == "bmm":
            q_device.densities = apply_unitary_density_bmm(density, matrix, wires)
    else:
        state = q_device.states
        if method == "einsum":
            func = apply_unitary_einsum
        elif method == "bmm":
            raise ValueError("In `gate_wrapper`: `bmm` not supported for `method`")
            func = apply_unitary_bmm
        
        # manually turn reals into complex
        states_real = func(state, matrix_real, wires)
        states_imag = func(state, matrix_imag, wires)
        states_imag_flipped = torch.einsum('ij,jk...->ik...',
            distribute_tensor(torch.Tensor([[0, -1], [1, 0]]), state.device_mesh, [Replicate()]),
            states_imag
        )
        q_device.states = states_real + states_imag_flipped

def rot(
    name,  # rx, ry, or rz
    q_device, wires, params=None, comp_method="einsum",
):
    mat = getattr(RS[name[-1]], f'{name}_matrix')
    gate_wrapper(
        name=name, mat=mat, method=comp_method, q_device=q_device,
        wires=wires, params=params,
    )

def pauli(
    name,  # (c)x, (c)y, or (c)z
    q_device, wires, params=None, comp_method="einsum",
):
    full_name = f'pauli{name}' if len(name) == 1 else name
    mat = getattr(PAULIS[name[-1]], f'_{name[-1]}_mat_dict')[full_name]
    gate_wrapper(
        name=name, mat=mat, method=comp_method, q_device=q_device,
        wires=wires, params=None,
    )

pauli_names = [a + b for a,b in itertools.product(['c', ''], ['x', 'y', 'z'])] 
rot_names = ['r' + b for b in ['x', 'y', 'z']]
# populate namespace
for name_ in pauli_names:
    vars()[name_] = partial(pauli, name_)
for name_ in rot_names:
    vars()[name_] = partial(rot, name_)

func_names = rot_names + pauli_names

# set all to einsum
for name_ in func_names:
    func_einsum = partialmethod(eval(name_), comp_method="einsum")
    setattr(DistributedQuantumDevice, name_, func_einsum)

class Op(torch.nn.Module):
    def __init__(self, func, wires, has_params=True, trainable=True, **unused):
        super().__init__()
        self.func_ = func
        self.wires = wires
        self.has_params = has_params
        self.trainable = trainable
        self.params = None
        if has_params:
            self.params = torch.empty(1)
            if trainable:
                self.params = torch.nn.Parameter(self.params)
    
    def forward(self, qdev, wires=None, params=None):
        self.func_(
            qdev,
            wires if wires is not None else self.wires,
            params=params if params is not None else self.params
        )

def OpFactory(name, has_params=True, trainable=True):
    """
    programattically creates RY from ry, CX from cx, etc
    `name` is lower case
    """
    def __init__(self, wires, **kwargs):
        kwargs.update({'has_params': kwargs.get('has_params', has_params)})
        kwargs.update({'trainable': kwargs.get('trainable', trainable)})
        Op.__init__(self, eval(name), wires, **kwargs)
    newclass = type(name.upper(), (Op, ), {"__init__": __init__})
    return newclass

for name_ in rot_names:
    vars()[name_.upper()] = OpFactory(name_)
for name_ in pauli_names:
    vars()[name_.upper()] = OpFactory(name_, has_params=False, trainable=False)