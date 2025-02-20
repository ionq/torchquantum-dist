import os

import numpy as np
import torch
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

def sampler_diff_approx(
    state_mag: DTensor, shots: int
) -> DTensor:
    # state_mag is a single state vector with no batch dimension
    rank = int(os.environ['RANK'])
    world_sz = int(os.environ['WORLD_SIZE'])
    p = state_mag
    # cheap good gaussian: https://stats.stackexchange.com/a/454431
    p_2 = torch.sqrt(p + 1e-16)  # eps to avoid nan in gradient at 0

    z = torch.randn(p.to_local().size(), device=p.device)
    e_d = torch.zeros(p.to_local().size(), device=p.device)
    if rank == world_sz - 1:
        z[-1] = 0
        e_d[-1] = 1
    z = DTensor.from_local(z, device_mesh=p.device_mesh, placements=p.placements)
    e_d = DTensor.from_local(e_d, device_mesh=p.device_mesh, placements=p.placements)

    # all-reduce
    u = (p_2 - e_d) / (torch.norm(p_2 - e_d) + 1e-8)
    mu = p * shots
    # Householder
    Qz = z - 2 * u * (z * u).sum()  # another all-reduce
    v = p_2 * Qz * (shots ** 0.5)  # should stay distributed

    # new counts, differentiably rectified to be positive only, normalized to sum to 1
    state_mag_noisy = torch.nn.functional.relu(v + mu)
    state_mag_noisy = state_mag_noisy / (state_mag_noisy.sum() + 1e-8)  # another all-reduce
    return state_mag_noisy

def sampler_nondiff_exact(
    state_mag: DTensor, shots: int
) -> DTensor:
    # state_mag is a single state vector with no batch dimension
    # hierarchical: first figure out N_i for i-th GPU, then within each GPU, sample N_i.
    # assumes all workers share a rng state
    orig_shape_local = state_mag.to_local().shape
    shard_dims = [s_.dim for s_ in state_mag.placements]
    reduce_dims = np.delete(list(range(state_mag.ndim)), shard_dims)
    gpu_probs = state_mag.sum(list(reduce_dims))  # shouldn't require comms
    gpu_probs = gpu_probs.full_tensor().ravel()  # all gather

    global_rank = int(os.environ['RANK'])
    local_shots = int(torch.distributions.multinomial.Multinomial(shots, gpu_probs).sample()[global_rank].item())

    if local_shots > 0:
        state_mag_local = state_mag.to_local().ravel()
        state_mag_local_norm = state_mag_local / (state_mag_local.sum() + 1e-8)
        state_mag_noisy  = torch.distributions.multinomial.Multinomial(local_shots, state_mag_local_norm).sample() / shots
        state_mag_noisy = state_mag_noisy.reshape(orig_shape_local)
    else:  # skip sampling and prevent nan
        state_mag_noisy = torch.zeros(orig_shape_local, device=state_mag.device)
    state_mag_noisy = DTensor.from_local(state_mag_noisy, device_mesh=state_mag.device_mesh, placements=state_mag.placements)
    return state_mag_noisy

def measure_allZ(
    q_device, shots: int=0, training: bool=False
):
    states = q_device.states
    # from here, work on a single state vector, no batch dimension
    state_mag = (states ** 2).sum(0)  # PauliZ hardocded here; no rotation before grabbing probabilities
    all_dims = np.arange(state_mag.dim())

    if shots > 0:
        if training:
            state_mag_noisy = sampler_diff_approx(state_mag, shots)
        else:
            torch.manual_seed(q_device.shared_seed)
            q_device.shared_seed += 1
            state_mag_noisy = sampler_nondiff_exact(state_mag, shots)
    else:
        state_mag_noisy = state_mag

    probs = []
    for wire in range(q_device.n_wires):
        reduction_dims = np.delete(all_dims, [wire])
        prob_ = state_mag_noisy.sum(list(reduction_dims))
        probs.append(prob_)
    probs = torch.stack(probs, dim=-2).full_tensor()  # all gather (q, 2)
    y = probs @ torch.tensor([1., -1.], device=probs.device)  # hardcoded PauliZ

    return y.unsqueeze(0)  # (1, q)
