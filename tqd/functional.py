import functools
from typing import Union

import numpy as np
import torch
from torch.autograd import Function, backward
from torch.distributed.tensor import DTensor

from .matrices import GATE_MAT_DICT

def maybe_to_local(tensor: torch.Tensor | DTensor) -> torch.Tensor | DTensor:
    """
    calls `.to_local()` if `tensor` is a `DTensor`, otherwise leaves it alone
    """
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor

class InvertibleUnitaryBMM(Function):
    """
    Implements an unitary batched matrix multiply as an invertible computation to save activation memory
    """
    @staticmethod
    def forward(ctx, matrix, state, dummy):
        # `dummy` allows passing activations thru backwards pass (in place of actual gradients)
        ctx.save_for_backward(matrix)
        return matrix.bmm(state), dummy

    @staticmethod
    def backward(ctx, gO, output):
        # Recompute input from output
        matrix, = ctx.saved_tensors
        inp = matrix.mH.bmm(output)
        del output

        inp = inp.detach().requires_grad_()
        mtx = matrix.detach().requires_grad_()
        with torch.enable_grad():
            out = mtx.bmm(inp)
        backward(out, gO)
        return mtx.grad, inp.grad, inp

class InvertiblePostUnitaryStep(Function):
    """
    No-op in forwards, but starts to pass the output back for the invertible computation
    """
    @staticmethod
    def forward(ctx, inp, dummy):
        # `dummy` allows passing activations thru backwards pass (in place of actual gradients)
        out = inp  # no-op, but easier to follow logic
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, gO):
        # Hijacking `dummy.grad` for activations here
        out, = ctx.saved_tensors
        return gO, out

def apply_unitary_bmm(
    state: Union[DTensor, torch.Tensor], mat: torch.Tensor,
    wires: Union[int, list[int]], wire_order: list[int],
    invertible_dummy: Union[DTensor, torch.Tensor]=None
):
    """
    Apply the unitary to the statevector using local batch matrix multiply.
    Note: Assumes that none of the sharding dimensions are affected by wires.

    Args:
        state (DTensor or torch.Tensor): The batched statevectors as a real DTensor. in last dim, 0th index is real part, 1st index is imaginary part
        mat (torch.Tensor): The batched unitary matrix of the operation as a complex Tensor.
        wires (int or List[int]): Which qubit(s) the operation is applied to.
        invertible_dummy (DTensor or torch.Tensor): use invertible computation to save memory? if yes, needs to be a dummy tensor to store gradients. if no, use None

    Returns:
        torch.Tensor: The new batch of statevectors.

    """
    mat = mat.to(state.device)
    gate_dims = [w + 1 for w in wires]

    pre = []
    post = []
    for gd_ in gate_dims:
        # order matters, so we search thru `gate_dims` in order
        pre.append(1 + wire_order.index(gd_ - 1))
    for i, w in enumerate(wire_order):
        if w + 1 not in gate_dims:
            #order doesn't matter
            post.append(i+1)
    permute_to = pre + post
    new_wire_order = [wire_order[d - 1] for d in permute_to]
    permute_to = [0] + permute_to + [state.dim()-1]
    orig_local_shape = maybe_to_local(state).shape
    permuted_local_shape = [orig_local_shape[i] for i in permute_to]
    bsz = orig_local_shape[0]
    permuted = state.permute(permute_to)
    is_dtensor = isinstance(permuted, DTensor)
    if is_dtensor:
        perm_dm, perm_place = permuted.device_mesh, permuted.placements
    permuted = torch.view_as_complex(maybe_to_local(permuted)).reshape([bsz, 2 ** len(wires), -1])
    if invertible_dummy is not None:
        invertible_dummy = invertible_dummy.permute(permute_to)
        invertible_dummy = torch.view_as_complex(maybe_to_local(invertible_dummy).contiguous()).reshape([bsz, 2 ** len(wires), -1])
    
    #permuted (b, m, k)
    #mat ([b,] n, m)
    if len(mat.shape) == 2:
        # matrix no batch, state in batch mode
        expand_shape = [bsz] + list(mat.shape)
        mat = mat.expand(expand_shape)
    # both matrix and state are in batch mode
    if invertible_dummy is not None:
        new_state, invertible_dummy = InvertibleUnitaryBMM.apply(mat, permuted, invertible_dummy)
    else:
        new_state = mat.bmm(permuted)
    if is_dtensor:
        new_state = DTensor.from_local(torch.view_as_real(new_state).view(permuted_local_shape), device_mesh=perm_dm, placements=perm_place)
        if invertible_dummy is not None:
            invertible_dummy = DTensor.from_local(torch.view_as_real(invertible_dummy).view(permuted_local_shape), device_mesh=perm_dm, placements=perm_place)
    else:
        new_state = torch.view_as_real(new_state).view(permuted_local_shape)
        if invertible_dummy is not None:
            invertible_dummy = torch.view_as_real(invertible_dummy).view(permuted_local_shape)

    return new_state, new_wire_order, invertible_dummy

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
            params = params.unsqueeze(-1).expand((q_device.bsz, -1))
        elif params.dim() == 0:
            params = params.unsqueeze(-1).unsqueeze(-1).expand((q_device.bsz, -1))
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

    state, wire_order, invertible_dummy = func(state, matrix, wires, wire_order, invertible_dummy=q_device._invertible_dummy)
    q_device._states = state
    q_device._wire_order = wire_order
    q_device._invertible_dummy = invertible_dummy

# populate namespace with functionals
for name_ in GATE_MAT_DICT.keys():
    vars()[name_] = functools.partial(gate, name_)
    vars()[f"{name_}_inv"] = functools.partial(gate, name_, inverse=True)

