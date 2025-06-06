import numpy as np
import torch
from torch.distributed.tensor import DTensor

def sampler_diff_approx(
    state_mag: DTensor, shots: int, global_rank: int, world_sz: int
) -> DTensor:
    # state_mag is a batch of state vectors
    p = state_mag
    # cheap good gaussian: https://stats.stackexchange.com/a/454431
    p_2 = torch.sqrt(p + 1e-16)  # eps to avoid nan in gradient at 0

    z = torch.randn(p.to_local().size(), device=p.device)
    e_d = torch.zeros(p.to_local().size(), device=p.device)
    if global_rank == world_sz - 1:
        z[-1] = 0
        e_d[-1] = 1
    z = DTensor.from_local(z, device_mesh=p.device_mesh, placements=p.placements)
    e_d = DTensor.from_local(e_d, device_mesh=p.device_mesh, placements=p.placements)

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
    state_mag: DTensor, shots: int, global_rank: int, *unused_args
) -> DTensor:
    # state_mag is a batch of state vectors
    # hierarchical: first figure out N_i for i-th GPU, then within each GPU, sample N_i.
    # assumes all workers share a rng state
    orig_shape_local = state_mag.to_local().shape
    shard_dims = [s_.dim for s_ in state_mag.placements if hasattr(s_, "dim")]
    reduce_dims = np.delete(list(range(state_mag.ndim)), [0] + shard_dims)
    gpu_probs = state_mag.sum(list(reduce_dims))  # shouldn't require comms
    gpu_probs = gpu_probs.full_tensor().view((orig_shape_local[0], -1))  # all gather  # (b, n_gpus)


    local_shots = torch.distributions.multinomial.Multinomial(shots, gpu_probs).sample()
    local_shots = local_shots[:, global_rank]

    state_mag_noisy = []
    state_mag_local = state_mag.to_local()
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
    state_mag_noisy = DTensor.from_local(state_mag_noisy, device_mesh=state_mag.device_mesh, placements=state_mag.placements)
    return state_mag_noisy

def measure_allZ(
    q_device, shots: int=0, training: bool=False
):
    states = q_device.states
    state_mag = (states ** 2).sum(-1)  # PauliZ hardocded here; no rotation before grabbing probabilities
    all_dims = np.arange(1, state_mag.dim())

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

    probs = []
    for wire in range(q_device.n_wires):
        reduction_dims = np.delete(all_dims, [wire])
        prob_ = state_mag_noisy.sum(list(reduction_dims))
        probs.append(prob_)
    probs = torch.stack(probs, dim=-2)
    if q_device.world_sz > 1:
        probs = probs.full_tensor()  # all gather (b, q, 2)
    y = probs @ torch.tensor([1., -1.], device=probs.device)  # hardcoded PauliZ

    return y  # (b, q)
