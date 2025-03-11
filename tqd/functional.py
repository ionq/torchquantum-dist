import functools
from typing import Union

import numpy as np
import torch
import torch.distributed
import torch.distributed.tensor
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from .matrices import GATE_MAT_DICT

def apply_unitary_bmm(state: DTensor, mat: torch.Tensor, wires: Union[int, list[int]], wire_order=None):
    """
    Apply the unitary to the statevector using manually implemented broadcasted matrix multiply
    in order to keep sharding semantics.
    Note: Assumes that none of the sharding dimensions are affected by wires.

    Args:
        state (DTensor): The statevector as a real DTensor. in last dim, 0th index is real part, 1st index is imaginary part
        mat (torch.Tensor): The unitary matrix of the operation as a complex Tensor.
        wires (int or List[int]): Which qubit(s) the operation is applied to.

    Returns:
        torch.Tensor: The new statevector.

    """

    gate_dims = [w + 1 for w in wires]

    permute_to = list(range(state.dim()-1))
    for d in sorted(gate_dims, reverse=True):
        del permute_to[d]
    permute_to = permute_to + gate_dims + [state.dim()-1]
    #TODO: remove this
    permute_back = list(np.argsort(permute_to))
    orig_shape = state.shape
    permuted = state.permute(permute_to).flatten(-len(wires)-1, -2)

    #permuted (b, ..., m, 2)
    #mat (n, m)
    new_state_local_complex = torch.einsum('ij,...j->...i', mat.to(permuted.device), torch.view_as_complex(permuted.to_local()))
    new_state = DTensor.from_local(torch.view_as_real(new_state_local_complex), device_mesh=state.device_mesh, placements=permuted.placements)
    # technically orig_shape is not quite right, but it's the same as the required shape
    # since it's all 2's except for batch size
    #TODO: remove the permute_back
    new_state = new_state.view(orig_shape).permute(permute_back)
    new_wire_order = list(range(state.ndim-2))

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

    # handle resharding here so that applying unitary on the state operates in parallel
    q_device.maybe_reshard(wires)

    state = q_device._states
    func = apply_unitary_bmm
    
    # manually turn reals into complex
    #print(f'{state} {matrix}')
    states, wire_order = func(state, matrix, wires, wire_order=q_device._wire_order)
    q_device._states = states
    q_device._wire_order = wire_order

# populate namespace with functionals
for name_ in GATE_MAT_DICT.keys():
    vars()[name_] = functools.partial(gate, name_)
