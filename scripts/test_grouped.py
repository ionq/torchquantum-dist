import os

import torch
import tqd


def test_grouped_gate_correctness(verbose=False):
    """
    Verify the grouped-qubit path (triggered when n_wires >= max_dtensor_dims - 2) produces
    the same statevector as the standard ungrouped path on the same circuit.
    Uses nq=7, max_dtensor_dims=7: qubits 1-2 in group 0 (dim 2), qubits 3-4 in group 1 (dim 3).
    """
    rank = os.environ["RANK"]
    nq = 7
    world_sz = 2

    qdev = tqd.DistributedQuantumDevice(
        nq, device="cuda", world_sz=world_sz, max_dtensor_dims=7
    )
    qdev_ref = tqd.DistributedQuantumDevice(
        nq, device="cuda", world_sz=world_sz, max_dtensor_dims=16
    )

    # Confirm grouping path is active
    assert (qdev._groupings[1] >= 0).any(), (
        "Expected grouped qubits for max_dtensor_dims=7, n_wires=7"
    )

    def run_circuit(dev):
        tqd.h(dev, wires=[0])
        dev.y(wires=[1])  # 1Q gate on grouped qubit (group 0, role 0)
        tqd.cx(dev, wires=[1, 4])  # cross-group CX
        tqd.rz(dev, wires=[3], params=torch.pi / 3)
        tqd.ry(dev, wires=[2], params=torch.pi / 4)
        tqd.cx(dev, wires=[3, 5])  # grouped → ungrouped CX (group 1, role 0 → q5)
        tqd.rz(dev, wires=[6], params=torch.pi / 5)

    run_circuit(qdev)
    run_circuit(qdev_ref)

    # Grouped and ungrouped layouts have different tensor shapes; flatten qubit dims before compare
    state_grouped = qdev.states.full_tensor().cpu().reshape(1, 2**nq, 2)
    state_ref = qdev_ref.states.full_tensor().cpu().reshape(1, 2**nq, 2)
    assert torch.allclose(state_grouped, state_ref, atol=1e-5), (
        f"Grouped statevector diverged from reference.\n"
        f"Max diff: {(state_grouped - state_ref).abs().max().item():.2e}"
    )
    if rank == "0":
        print("grouped gate correctness test passed!")


def test_restore_loop_regression(verbose=False):
    """
    Regression for bug 1: the restore loop in apply_unitary_bmm used stale grouping, causing
    silent wrong-qubit operations on 2Q gates involving grouped qubits. Exercises the restore
    loop through multiple CX gates across and within groups in both wire orderings, then
    verifies the final statevector against the ungrouped reference.
    """
    rank = os.environ["RANK"]
    nq = 7
    world_sz = 2

    qdev = tqd.DistributedQuantumDevice(
        nq, device="cuda", world_sz=world_sz, max_dtensor_dims=7
    )
    qdev_ref = tqd.DistributedQuantumDevice(
        nq, device="cuda", world_sz=world_sz, max_dtensor_dims=16
    )

    def run_circuit(dev):
        tqd.h(dev, wires=[1])
        tqd.h(dev, wires=[3])
        tqd.cx(dev, wires=[1, 4])  # group 0 → group 1
        tqd.cx(dev, wires=[4, 1])  # reversed: group 1 → group 0
        tqd.cx(dev, wires=[3, 2])  # group 1 → group 0, reversed-index order
        tqd.cx(dev, wires=[2, 3])  # group 0 → group 1
        tqd.rz(dev, wires=[1], params=torch.pi / 5)
        tqd.cx(dev, wires=[0, 2])  # ungrouped → grouped
        tqd.cx(dev, wires=[4, 5])  # grouped → ungrouped

    run_circuit(qdev)
    run_circuit(qdev_ref)

    state_grouped = qdev.states.full_tensor().cpu().reshape(1, 2**nq, 2)
    state_ref = qdev_ref.states.full_tensor().cpu().reshape(1, 2**nq, 2)
    assert torch.allclose(state_grouped, state_ref, atol=1e-5), (
        f"Restore loop regression: grouped statevector diverged.\n"
        f"Max diff: {(state_grouped - state_ref).abs().max().item():.2e}"
    )
    if rank == "0":
        print("restore loop regression test passed!")


def test_grouped_postselection(verbose=False):
    """
    Regression for bug 3: grouped-qubit postselection used the wrong index into the reshaped
    group dimension, causing IndexError. Uses nq=7, max_dtensor_dims=7 (2-qubit groups).
    Verifies no crash, correct output shape, and correct <Z> for the postselected wire.
    """
    rank = os.environ["RANK"]
    nq = 7
    world_sz = 2

    qdev = tqd.DistributedQuantumDevice(
        nq, device="cuda", world_sz=world_sz, max_dtensor_dims=7
    )

    grouped_wires = torch.nonzero(qdev._groupings[1] >= 0).flatten().tolist()
    assert len(grouped_wires) > 0, (
        "Expected grouped qubits for max_dtensor_dims=7, n_wires=7"
    )

    tqd.h(qdev, wires=[0])
    tqd.ry(qdev, wires=[grouped_wires[0]], params=torch.pi / 3)

    post_wire = grouped_wires[0]
    # Postselect on a grouped qubit; old code IndexError'd here (bug 3)
    meas, retained = tqd.measure_allZ(qdev, shots=0, postselect_cond={post_wire: 0})

    assert meas.shape == (1, nq), f"Unexpected meas shape: {meas.shape}"
    # retained is a DTensor when world_sz > 1; gather before indexing
    retained_cpu = retained.full_tensor().cpu()
    assert retained_cpu.shape == (1,), (
        f"Unexpected retained shape: {retained_cpu.shape}"
    )
    assert retained_cpu[0].item(), "Expected batch element retained after postselection"
    assert abs(meas[0, post_wire].item() - 1.0) < 1e-5, (
        f"Expected <Z>=1.0 for postselected wire {post_wire}, got {meas[0, post_wire].item():.6f}"
    )
    if rank == "0":
        print("grouped postselection test passed!")


if __name__ == "__main__":
    test_grouped_gate_correctness(False)
    test_restore_loop_regression(False)
    test_grouped_postselection(False)
