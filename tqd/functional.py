import functools
import importlib
import itertools
from typing import Callable

import numpy as np
import torch
import torch.distributed
import torch.distributed.tensor
from torch.distributed.tensor import Replicate, Shard, distribute_tensor
from torchquantum.macro import ABC, ABC_ARRAY, F_DTYPE

XYZ = ['x', 'y', 'z']

# modules from torchquantum
TQ_PAULIS = {}
TQ_RS = {}
for letter_ in XYZ:
    TQ_PAULIS[letter_] = importlib.import_module(f'.pauli{letter_}', f'torchquantum.functional')
    TQ_RS[letter_] = importlib.import_module(f'.r{letter_}', f'torchquantum.functional')

PAULI_NAMES = [a + b for a,b in itertools.product(['c', ''], XYZ)] 
ROT_NAMES = ['r' + b for b in XYZ]

# list of all the lower case functions we've ported
FUNC_NAMES = ROT_NAMES + PAULI_NAMES

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

    #TODO: do something smarter here
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
    mat = getattr(TQ_RS[name[-1]], f'{name}_matrix')
    gate_wrapper(
        name=name, mat=mat, method=comp_method, q_device=q_device,
        wires=wires, params=params,
    )

def pauli(
    name,  # (c)x, (c)y, or (c)z
    q_device, wires, params=None, comp_method="einsum",
):
    full_name = f'pauli{name}' if len(name) == 1 else name
    mat = getattr(TQ_PAULIS[name[-1]], f'_{name[-1]}_mat_dict')[full_name]
    gate_wrapper(
        name=name, mat=mat, method=comp_method, q_device=q_device,
        wires=wires, params=None,
    )

# populate namespace with functionals
for name_ in PAULI_NAMES:
    vars()[name_] = functools.partial(pauli, name_)
for name_ in ROT_NAMES:
    vars()[name_] = functools.partial(rot, name_)
