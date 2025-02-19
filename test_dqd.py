import os

import torch
import torchquantum as tq
import tqd
from torch.distributed.tensor import distribute_tensor, Replicate, Shard


def test_dqd(verbose=False):
    rank = os.environ['LOCAL_RANK']
    nq = 3
    world_sz = 2
    wire = 0

    qdev = tqd.DistributedQuantumDevice(
        nq,
        device=f'cuda',
        world_sz=world_sz
    )
    if verbose:
        print(f'before {rank} {qdev.states}')

    # test class method
    qdev.y(wires=[wire])

    # test Module
    cx_gate = tqd.CX(wires=[wire, (wire+1) % nq])
    cx_gate(qdev)
    
    # test functional
    tqd.rz(qdev, wires=[wire], params=torch.pi/3)

    # test registration
    tqd.custom.register_gate('i', torch.eye(2, dtype=torch.cfloat), False)
    tqd.custom.i(qdev, wires=[1])

    if verbose:
        print(f'after {rank} {qdev.states}')
        print(f'done {qdev.states.full_tensor()}')


    # compare against ground truth
    qdev_tq = tq.QuantumDevice(3)
    qdev_tq.y(wires=[wire])
    qdev_tq.cx(wires=[wire, (wire+1) % nq])
    qdev_tq.rz(wires=[wire], params=torch.pi/3)

    # remove singleton batch dimension and put complex split dimension in front to match our implementation
    states_tq = torch.view_as_real(qdev_tq.states).permute([0,4,1,2,3])[0]
    if verbose:
        print(f'torchquantum {states_tq}')

    assert(torch.allclose(states_tq, qdev.states.full_tensor().cpu()))

    if rank == '0':
        print('test passed!')

if __name__ == "__main__":
    test_dqd()