import functools
from typing import Union

import numpy as np
import torch
import torch.distributed
import torch.distributed.tensor
from torch.distributed.tensor import DTensor

from .matrices import GATE_MAT_DICT

def apply_unitary_bmm(state: DTensor, mat: torch.Tensor, wires: Union[int, list[int]], wire_order: list[int]):
    """
    Apply the unitary to the statevector using local batch matrix multiply.
    Note: Assumes that none of the sharding dimensions are affected by wires.

    Args:
        state (DTensor): The batched statevectors as a real DTensor. in last dim, 0th index is real part, 1st index is imaginary part
        mat (torch.Tensor): The batched unitary matrix of the operation as a complex Tensor.
        wires (int or List[int]): Which qubit(s) the operation is applied to.

    Returns:
        torch.Tensor: The new batch of statevectors.

    """
    mat = mat.to(state.device)
    gate_dims = [w + 1 for w in wires]

    pre = []
    post = []
    for i, w in enumerate(wire_order):
        if w + 1 in gate_dims:
            pre.append(i+1)
        else:
            post.append(i+1)
    permute_to = pre + post
    new_wire_order = [wire_order[d - 1] for d in permute_to]
    permute_to = [0] + permute_to + [state.dim()-1]
    orig_local_shape = state.to_local().shape
    permuted_local_shape = [orig_local_shape[i] for i in permute_to]
    bsz = orig_local_shape[0]
    permuted = state.permute(permute_to)
    perm_dm, perm_place = permuted.device_mesh, permuted.placements
    permuted = torch.view_as_complex(permuted.to_local()).reshape([bsz, 2 ** len(wires), -1])

    #permuted (b, m, k)
    #mat ([b,] n, m)
    if len(mat.shape) > 2:
        # both matrix and state are in batch mode
        new_state = mat.bmm(permuted)
    else:
        # matrix no batch, state in batch mode
        expand_shape = [bsz] + list(mat.shape)
        new_state = mat.expand(expand_shape).bmm(permuted)
    # technically orig_local_shape is not quite right, but it's the same as the required shape
    # since it's all 2's except for batch size (1's where sharded)
    new_state = DTensor.from_local(torch.view_as_real(new_state).view(permuted_local_shape), device_mesh=perm_dm, placements=perm_place)

    return new_state, new_wire_order

def gate(
    name, q_device, wires,
    params=None, inverse=False
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

    if inverse:
        matrix = matrix.mH

    assert np.log2(matrix.shape[-1]) == len(wires)

    # handle resharding here so that applying unitary on the state operates in parallel
    q_device.maybe_reshard(wires)

    state = q_device._states
    wire_order = q_device._wire_order
    func = apply_unitary_bmm

    state, wire_order = func(state, matrix, wires, wire_order)
    q_device._states = state
    q_device._wire_order = wire_order

# populate namespace with functionals
for name_ in GATE_MAT_DICT.keys():
    vars()[name_] = functools.partial(gate, name_)
    vars()[f"{name_}_inv"] = functools.partial(gate, name_, inverse=True)

