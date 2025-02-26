import functools
from typing import Union

import numpy as np
import torch
import torch.distributed
import torch.distributed.tensor
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from .matrices import GATE_MAT_DICT

def apply_unitary_bmm(state: DTensor, mat: torch.Tensor, wires: Union[int, list[int]]):
    """
    Apply the unitary to the statevector using manually implemented broadcasted matrix multiply
    in order to keep sharding semantics.
    Note: Assumes that none of the sharding dimensions are affected by wires.

    Args:
        state (DTensor): The statevector as a real DTensor. 0th index is real part, 1st index is imaginary part
        mat (torch.Tensor): The real or imaginary component of a unitary matrix of the operation as a real Tensor.
        wires (int or List[int]): Which qubit(s) the operation is applied to.

    Returns:
        torch.Tensor: The new statevector.

    """

    gate_dims = [w + 1 for w in wires]
    mat = distribute_tensor(mat, state.device_mesh, [Replicate()])

    permute_to = list(range(state.dim()))
    for d in sorted(gate_dims, reverse=True):
        del permute_to[d]
    permute_to = permute_to + gate_dims
    #TODO: remove this
    permute_back = list(np.argsort(permute_to))
    orig_shape = state.shape
    permuted = state.permute(permute_to).flatten(-len(wires), -1)

    #permuted (b, ..., p, m)
    #mat (n, m)
    new_state = (mat * permuted.unsqueeze(-2)).sum(-1)

    # technically orig_shape is not quite right, but it's the same as the required shape
    # since it's all 2's except for batch size
    #TODO: remove the permute_back
    new_state = new_state.view(orig_shape).permute(permute_back)
    new_wire_order = list(range(state.ndim-1))

    return new_state, new_wire_order

def gate(
    name, q_device, wires,
    params=None
):
    mat = GATE_MAT_DICT[name]
    if params is not None:
        if not isinstance(params, torch.Tensor):
            # this is for directly inputting parameters as a number
            params = torch.tensor(params, dtype=torch.float32)

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
    if callable(mat):
        matrix = mat(params)
    else:
        matrix = mat

    assert np.log2(matrix.shape[-1]) == len(wires)

    matrix_real, matrix_imag = (lambda x: (x[..., 0], x[..., 1]))(torch.view_as_real(matrix))
    # handle resharding logic here so that applying unitary on the state operates in parallel
    q_device.maybe_reshard(wires)

    state = q_device._states
    func = apply_unitary_bmm
    
    # manually turn reals into complex
    states_real, _ = func(state, matrix_real, wires)
    states_imag, wire_order = func(state, matrix_imag, wires)
    states_imag_flipped = torch.einsum('ij,jk...->ik...',
        distribute_tensor(torch.Tensor([[0, -1], [1, 0]]), state.device_mesh, [Replicate()]),
        states_imag
    )
    q_device._states = states_real + states_imag_flipped
    q_device._wire_order = wire_order

# populate namespace with functionals
for name_ in GATE_MAT_DICT.keys():
    vars()[name_] = functools.partial(gate, name_)
