import numpy as np
import torch
from typing import Union
from torch.distributed.tensor import DTensor

from .utils.maybe_dtensor import is_dtensor, maybe_to_local, maybe_get_dtensor_info, maybe_from_local, maybe_full_tensor

def sampler_diff_approx(
    state_mag: Union[torch.Tensor, DTensor], shots: int, global_rank: int, world_sz: int
) -> Union[torch.Tensor, DTensor]:
    # state_mag is a batch of state vectors
    p = state_mag
    # cheap good gaussian: https://stats.stackexchange.com/a/454431
    p_2 = torch.sqrt(p + 1e-16)  # eps to avoid nan in gradient at 0

    z = torch.randn(maybe_to_local(p).size(), device=p.device)
    e_d = torch.zeros(maybe_to_local(p).size(), device=p.device)
    if global_rank == world_sz - 1:
        z[-1] = 0
        e_d[-1] = 1
    p_device_mesh, p_placements = maybe_get_dtensor_info(p)
    z = maybe_from_local(z, device_mesh=p_device_mesh, placements=p_placements)
    e_d = maybe_from_local(e_d, device_mesh=p_device_mesh, placements=p_placements)

    # all-reduce
    reduce_dims = tuple(range(1, p.ndim))
    u = (p_2 - e_d) / (torch.linalg.vector_norm(p_2 - e_d, dim=reduce_dims, keepdim=True) + 1e-8)
    mu = p * shots
    # Householder
    Qz = z - 2 * u * (z * u).sum(reduce_dims, keepdim=True)  # another all-reduce
    v = p_2 * Qz * (shots ** 0.5)  # should stay distributed

    # new counts, differentiably rectified to be positive only, normalized to sum to 1
    state_mag_noisy = torch.nn.functional.relu(v + mu)
    state_mag_noisy = state_mag_noisy / (state_mag_noisy.sum(reduce_dims, keepdim=True) + 1e-8)  # another all-reduce
    return state_mag_noisy

def sampler_nondiff_exact(
    state_mag: Union[torch.Tensor, DTensor], shots: int, global_rank: int, *unused_args
) -> Union[torch.Tensor, DTensor]:
    # state_mag is a batch of state vectors
    # hierarchical: first figure out N_i for i-th GPU, then within each GPU, sample N_i.
    # assumes all workers share a rng state
    orig_shape_local = maybe_to_local(state_mag).shape
    state_mag_device_mesh, state_mag_placements = maybe_get_dtensor_info(state_mag)
    shard_dims = [s_.dim for s_ in state_mag_placements if hasattr(s_, "dim")]
    reduce_dims = np.delete(list(range(state_mag.ndim)), [0] + shard_dims)
    gpu_probs = state_mag.sum(list(reduce_dims))  # shouldn't require comms
    gpu_probs = maybe_full_tensor(gpu_probs).view((orig_shape_local[0], -1))  # all gather  # (b, n_gpus)

    if is_dtensor(state_mag):
        local_shots = torch.distributions.multinomial.Multinomial(shots, gpu_probs).sample()
        local_shots = local_shots[:, global_rank]
    else:
        local_shots = shots*torch.ones(orig_shape_local[0], device=state_mag.device)

    state_mag_noisy = []
    state_mag_local = maybe_to_local(state_mag)
    # TODO: seems multinomial sampling isn't vectorized
    for i in range(orig_shape_local[0]):
        local_shots_ = int(local_shots[i].item())
        if local_shots_ > 0:
            state_mag_local_ = state_mag_local[i].view(-1)

            state_mag_local_norm_ = state_mag_local_ / (state_mag_local_.sum() + 1e-8)
            state_mag_noisy_ = torch.distributions.multinomial.Multinomial(local_shots_, state_mag_local_norm_).sample() / shots
            state_mag_noisy_ = state_mag_noisy_.reshape(orig_shape_local[1:])
        else:  # skip sampling and prevent nan
            state_mag_noisy_ = torch.zeros(orig_shape_local[1:], device=state_mag.device)
        state_mag_noisy.append(state_mag_noisy_)
    state_mag_noisy = torch.stack(state_mag_noisy)
    state_mag_noisy = maybe_from_local(state_mag_noisy, device_mesh=state_mag_device_mesh, placements=state_mag_placements)
    return state_mag_noisy

def measure_allZ(
        q_device, shots: int=0, postselect_cond: dict[int, int]={}, training: bool=False
):
    states, groupings  = q_device.noncanonical_states
    sharded_wires = torch.nonzero(groupings[1] == -2).flatten()
    grouped_wires = torch.nonzero(groupings[1] >= 0).flatten()
    ungrouped_wires = torch.nonzero(groupings[1] == -1).flatten()
    
    state_mag = (states ** 2).sum(-1)  # PauliZ hardocded here; no rotation before grabbing probabilities

    # postselect_cond dictionary determines whether to ignore slices of state_mag (i.e. {0: 1} indicates wire zero should be |1>)
    if postselect_cond:
        assert set(postselect_cond.values()).issubset({0,1})
        local_mask_size = maybe_to_local(state_mag).size()
        local_mask_size = [local_mask_size[i] if i > 0 else 1 for i in range(len(local_mask_size))]
        local_mask = torch.ones(local_mask_size, device=states.device, dtype=bool)
        states_device_mesh, states_placements = maybe_get_dtensor_info(states)
        post_wires = list(postselect_cond.keys())
        post_bits = list(postselect_cond.values())
        post_groups = groupings[:, post_wires]
        for i in range(len(post_wires)):
            wire_info = post_groups[:, i].flatten()
            if wire_info[1].item() in [-1, -2]: # Ungrouped/sharded qubits are easiest to handle
                slice_idx = (slice(None), )*wire_info[0].item() + (1 - post_bits[i], ) + (slice(None), )*(len(local_mask_size) - wire_info[0].item() - 1)
            else: # Grouped qubits require more delicate manipulation of the statevector
                group_size = state_mag.shape[wire_info[0].item()]
                group_bool = torch.zeros(group_size, device=local_mask.device, dtype=bool).reshape((2,)*int(np.log(group_size)/np.log(2)))
                # For correct relative dimension, set nonselected dimension to False
                group_bool[(slice(None), )*wire_info[1].item() + (1 - post_bits[i], ) + (slice(None), )*(group_size - wire_info[1].item() - 1)] = True
                slice_idx = (slice(None), )*wire_info[0].item() + (group_bool.flatten(), ) + (slice(None), )*(len(local_mask_size) - wire_info[0].item() - 1)
            local_mask[slice_idx] = False

        full_mask = maybe_from_local(local_mask, device_mesh=states_device_mesh, placements=states_placements)
        expanded_mask = full_mask.expand((local_mask_size[0], ) + (-1, ) * (full_mask.ndim - 1))
        state_mag = torch.where(full_mask, state_mag, torch.zeros_like(state_mag))
        norm_square = state_mag.sum([d for d in range(full_mask.ndim) if d > 0]).reshape(-1, 1)
        retained = (norm_square.squeeze(-1) > 1e-16)
        if not torch.any(retained):
            raise RuntimeError('Postselection did not retain any batch elements.')
        elif not torch.all(retained):
            state_mag = state_mag[retained]
            norm_square = norm_square[retained]
        state_mag = state_mag/norm_square
    
    if shots > 0:
        if not training:
            torch.manual_seed(q_device.shared_seed)
            q_device.shared_seed += 1
            sampler = sampler_nondiff_exact
        else:
            sampler = sampler_diff_approx
    else:  # no noise; identity w/ extra args
        sampler = lambda x, _0, _1, _2: x
    state_mag_noisy = sampler(state_mag, shots, q_device.global_rank, q_device.world_sz)

    probs = torch.zeros((state_mag_noisy.shape[0], q_device.n_wires, 2), device=state_mag_noisy.device)
    # First reduce along sharded dimensions, then calculate probs for unsharded qubits
    sharded_reduce_list = list(groupings[0,sharded_wires])
    if sharded_reduce_list:
        shard_reduced_state_mag = state_mag_noisy.sum(list(groupings[0,sharded_wires]))
    else:
        shard_reduced_state_mag = state_mag_noisy
    remaining_dims = torch.arange(1, shard_reduced_state_mag.ndim, device=state_mag_noisy.device)
    for wire in ungrouped_wires:
        reduce_list = remaining_dims[remaining_dims != groupings[0,wire].item()].tolist()
        if reduce_list:
            prob_ = shard_reduced_state_mag.sum(reduce_list)
        else:
            prob_ = shard_reduced_state_mag
        prob_ = maybe_full_tensor(prob_)
        probs[:,wire,:] = prob_

    # pick ungrouped wire to interchange with all grouped wires
    prev_wire = ungrouped_wires[0]
    reduction_dims = remaining_dims[remaining_dims != groupings[0, prev_wire].item()].tolist() # Unreduced dim remains the same as qubits are shunted around
    shard_reduced_groupings = groupings.detach().clone()
    for wire in grouped_wires:
        shard_reduced_state_mag, shard_reduced_groupings = q_device.interchange_qubits(shard_reduced_state_mag, shard_reduced_groupings, wire, prev_wire)
        prob_ = shard_reduced_state_mag.sum(reduction_dims)
        if q_device.world_sz > 1:
            prob_ = prob_.full_tensor()
        probs[:,wire,:] = prob_
        prev_wire = wire

    # Then reduce unsharded dimensions and calculate probs for sharded qubits
    if is_dtensor(state_mag_noisy):
        remaining_dims = torch.arange(1, state_mag_noisy.ndim, device=state_mag_noisy.device)
        unshard_mask = torch.ones(remaining_dims.shape, dtype=bool, device=remaining_dims.device)
        for wire in sharded_wires:
            unshard_mask &= (remaining_dims != groupings[0,wire].item())
        only_shard_state_mag = state_mag_noisy.sum(remaining_dims[unshard_mask].tolist())
        remaining_dims = torch.arange(1, q_device.log2_devices + 1, device=remaining_dims.device)
        only_shard_groupings = groupings.detach().clone()
        only_shard_groupings[0, sharded_wires] -= min(only_shard_groupings[0, sharded_wires]) - 1 # reindex sharded dimensions for reduced tensor
        for wire in sharded_wires:
            reduce_list = remaining_dims[remaining_dims != only_shard_groupings[0,wire].item()].tolist()
            if reduce_list:
                prob_ = only_shard_state_mag.sum(reduce_list)
            else:
                prob_ = only_shard_state_mag
            if q_device.world_sz > 1:
                prob_ = prob_.full_tensor()
            probs[:,wire,:] = prob_

    y = probs @ torch.tensor([1., -1.], device=probs.device)  # hardcoded PauliZ

    # return y, retained if postselect_cond else y # (b, q)
    if postselect_cond:
        result = (y, retained)
    else:
        result = (y)
    return result # (b, q)
