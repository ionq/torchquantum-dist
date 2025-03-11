import os

import torch
import tqd
from torch.distributed.tensor import distribute_tensor, Replicate, Shard


def test_dqd(verbose=False):
    """
    monotest
    """
    rank = os.environ['RANK']
    nq = 3
    world_sz = 2
    wire = 1

    qdev = tqd.DistributedQuantumDevice(
        nq,
        device=f'cuda',
        world_sz=world_sz
    )
    if verbose:
        print(f'before {rank} {qdev.states}')

    # test class method
    qdev.y(wires=[wire])
    if verbose:
        print(f'after y {rank} {qdev.states}')

    # test Module
    cx_gate = tqd.CX(wires=[wire, (wire+1) % nq])
    cx_gate(qdev)
    if verbose:
        print(f'after cx {rank} {qdev.states}')

    # test functional
    tqd.rz(qdev, wires=[wire], params=torch.pi/3)
    if verbose:
        print(f'after rz {rank} {qdev.states}')

    # test registration
    tqd.custom.register_gate('i', torch.eye(2, dtype=torch.cfloat))
    tqd.custom.i(qdev, wires=[1])

    if verbose:
        print(f'after {rank} {qdev.states}')
        print(f'done {qdev.states.full_tensor()}')

    """
    # compare against ground truth
    qdev_tq = tq.QuantumDevice(3)
    qdev_tq.y(wires=[wire])
    qdev_tq.cx(wires=[wire, (wire+1) % nq])
    qdev_tq.rz(wires=[wire], params=torch.pi/3)

    # remove singleton batch dimension and put complex split dimension in front to match our implementation
    states_tq = torch.view_as_real(qdev_tq.states).permute([0,4,1,2,3])[0]
    """
    states_tq = torch.Tensor([[[[ 0.,  0.],
          [ 0.,  0.]],
         [[ 0.,  0.],
          [-0.5, torch.sqrt(torch.tensor([3]))/2]]],
        [[[ 0.,  0.],
          [ 0.,  0.]],
         [[ 0.,  0.],
          [ 0.,  0.]]]])

    if verbose:
        print(f'torchquantum {states_tq}')
    assert(torch.allclose(states_tq, qdev.states.full_tensor().cpu()[0]))
    if rank == '0':
        print('class method, module, functional, registration, and correctness test passed!')

    # test noiseless measurement
    meas = tqd.measure_allZ(qdev)
    """
    meas_tq = tq.MeasureAll(tq.PauliZ)(qdev_tq)
    """
    meas_tq = torch.Tensor([[ 1., -1., -1.]])
    if verbose:
        print(f'meas {rank} {meas} {meas_tq}')
    assert(torch.allclose(meas_tq, meas.cpu()))
    if rank == '0':
        print('noiseless measurement passed!')

    # test noisy non-differentiable (sampling) measurement
    meas_samp = tqd.measure_allZ(qdev, shots=1024, training=False)
    if verbose:
        print(f'meas_noisy non-diff {rank} {meas_samp}')
    assert(torch.allclose(meas_tq, meas_samp.cpu()))

    # test noisy differentiable (approximate) measurement
    meas_approx = tqd.measure_allZ(qdev, shots=1024, training=True)
    if verbose:
        print(f'meas_noisy diff {rank} {meas_approx}')
    assert(torch.allclose(meas_tq, meas_approx.cpu()))

    if rank == '0':
        print('monotest noisy measurement test passed!')

def test_noisy_meas(verbose=False):
    rank = os.environ['RANK']
    nq = 3
    world_sz = 2
    wire = 1

    qdev = tqd.DistributedQuantumDevice(
        nq,
        device=f'cuda',
        world_sz=world_sz
    )
    qdev.rx(wires=[wire], params=torch.pi/6)

    """
    qdev_tq = tq.QuantumDevice(3)
    qdev_tq.rx(wires=[wire], params=torch.pi/6)
    meas_tq = tq.MeasureAll(tq.PauliZ)(qdev_tq)
    """
    meas_tq = torch.Tensor([[1., torch.sqrt(torch.tensor([3]))/2, 1.]])

    # test noisy non-differentiable (sampling) measurement
    meas_samp = tqd.measure_allZ(qdev, shots=int(1e5), training=False)

    # test noisy differentiable (approximate) measurement
    meas_approx = tqd.measure_allZ(qdev, shots=int(1e5), training=True)
    if verbose:
        print(f'meas_noisy {rank} {meas_tq} {meas_samp} {meas_approx}')

    assert(torch.allclose(meas_tq, meas_approx.cpu(), rtol=1e-3, atol=1e-2))
    assert(torch.allclose(meas_tq, meas_samp.cpu(), rtol=1e-3, atol=1e-2))

    if rank == '0':
        print('standalone noisy measurement test passed!')

if __name__ == "__main__":
    test_dqd(False)
    test_noisy_meas(False)