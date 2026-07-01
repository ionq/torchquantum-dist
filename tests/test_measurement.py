import os
import math

import torch
import tqd


def test_noiseless_measurement_values(verbose=False):
    """
    Apply Ry(theta) on wire 0 and verify noiseless measure_allZ returns cos(theta) for
    that wire and 1.0 for all others (still in |0>).
    """
    rank = os.environ["RANK"]
    nq = 3
    world_sz = 2
    theta = math.pi / 3

    qdev = tqd.DistributedQuantumDevice(nq, device="cuda", world_sz=world_sz)
    tqd.ry(qdev, wires=[0], params=theta)

    meas = tqd.measure_allZ(qdev, shots=0)
    expected = torch.Tensor([[math.cos(theta), 1.0, 1.0]])

    assert torch.allclose(meas.cpu(), expected, atol=1e-5), (
        f"Expected {expected}, got {meas.cpu()}"
    )
    if rank == "0":
        print("noiseless measurement values test passed!")


def test_measurement_after_reshard(verbose=False):
    """
    Verify that measure_allZ returns analytically correct <Z> values after a CX gate
    has triggered resharding (CX on [1, 2] where wire 2 is initially sharded).
    """
    rank = os.environ["RANK"]
    nq = 3
    world_sz = 2

    qdev = tqd.DistributedQuantumDevice(nq, device="cuda", world_sz=world_sz)
    tqd.ry(qdev, wires=[0], params=math.pi / 3)
    tqd.ry(qdev, wires=[1], params=math.pi / 4)
    tqd.cx(qdev, wires=[1, 2])  # q2 is initially sharded; this triggers resharding

    meas = tqd.measure_allZ(qdev, shots=0)
    # Circuit yields state: sum over |q0>|q1,q2=00 or 11>
    # <Z>_0 = cos(pi/3) = 0.5
    # <Z>_1 = cos^2(pi/8) - sin^2(pi/8) = cos(pi/4)
    # <Z>_2 = same as <Z>_1 (q2 mirrors q1 via CX)
    expected = torch.Tensor(
        [[math.cos(math.pi / 3), math.cos(math.pi / 4), math.cos(math.pi / 4)]]
    )

    assert torch.allclose(meas.cpu(), expected, atol=1e-5), (
        f"Expected {expected}, got {meas.cpu()}"
    )
    if rank == "0":
        print("measurement after reshard test passed!")


def test_cuda_rng_seeding(verbose=False):
    """
    Regression for bug 4: torch.manual_seed (CPU) was used instead of torch.cuda.manual_seed,
    making shot-based sampling non-deterministic across runs with the same shared_seed.
    Two fresh devices with the same shared_seed and identical gates must produce identical samples.
    """
    rank = os.environ["RANK"]
    nq = 3
    world_sz = 2
    seed = 42

    qdev_a = tqd.DistributedQuantumDevice(
        nq, device="cuda", world_sz=world_sz, shared_seed=seed
    )
    qdev_b = tqd.DistributedQuantumDevice(
        nq, device="cuda", world_sz=world_sz, shared_seed=seed
    )

    for dev in [qdev_a, qdev_b]:
        tqd.ry(dev, wires=[0], params=math.pi / 3)
        tqd.ry(dev, wires=[1], params=math.pi / 4)

    meas_a = tqd.measure_allZ(qdev_a, shots=1000, training=False)
    meas_b = tqd.measure_allZ(qdev_b, shots=1000, training=False)

    assert torch.allclose(meas_a.cpu(), meas_b.cpu()), (
        f"Sampling results differ despite same shared_seed={seed}:\n"
        f"  a: {meas_a.cpu()}\n  b: {meas_b.cpu()}"
    )
    if rank == "0":
        print("CUDA RNG seeding test passed!")


def test_postselect_normalization(verbose=False):
    """
    After postselecting wire 0 to |0>, the collapsed state should be |000>, giving <Z>=1.0
    for every wire. Also checks that the retained flag is set.
    """
    rank = os.environ["RANK"]
    nq = 3
    world_sz = 2

    qdev = tqd.DistributedQuantumDevice(nq, device="cuda", world_sz=world_sz)
    # Ry(pi/3) puts wire 0 in a superposition; postselecting |0> collapses to |000>
    tqd.ry(qdev, wires=[0], params=math.pi / 3)

    meas, retained = tqd.measure_allZ(qdev, shots=0, postselect_cond={0: 0})

    expected = torch.Tensor([[1.0, 1.0, 1.0]])
    assert torch.allclose(meas.cpu(), expected, atol=1e-5), (
        f"Expected {expected} after postselection, got {meas.cpu()}"
    )
    # retained is a DTensor when world_sz > 1; gather before indexing
    retained_cpu = retained.full_tensor().cpu()
    assert retained_cpu[0].item(), "Expected batch element retained after postselection"
    if rank == "0":
        print("postselect normalization test passed!")


if __name__ == "__main__":
    test_noiseless_measurement_values(False)
    test_measurement_after_reshard(False)
    test_cuda_rng_seeding(False)
    test_postselect_normalization(False)
