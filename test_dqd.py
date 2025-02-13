import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torchquantum as tq
from torch.distributed.tensor import distribute_tensor, Replicate, Shard

import tqd
from tqd import DistributedQuantumDevice

def test_dqd(verbose=False):
    rank = os.environ['LOCAL_RANK']
    nq = 3
    world_sz = 2
    qdev = DistributedQuantumDevice(
        nq,
        device=f'cuda',
        world_sz=world_sz
    )
    if verbose:
        print(f'before {rank} {qdev.states}')

    # test class method
    qdev.y(wires=[2])

    # test Module
    x_gate = tqd.X(wires=[2])
    x_gate(qdev)
    
    # test functional
    tqd.rz(qdev, wires=[2], params=torch.pi/3)
    
    if verbose:
        print(f'after {rank} {qdev.states}')
        print(f'done {qdev.states.full_tensor()}')


    # compare against ground truth
    qdev_tq = tq.QuantumDevice(3)
    qdev_tq.y(wires=[2])
    qdev_tq.x(wires=[2])
    qdev_tq.rz(wires=[2], params=torch.pi/3)

    # remove singleton batch dimension and put complex split dimension in front to match our implementation
    states_tq = torch.view_as_real(qdev_tq.states).permute([0,4,1,2,3])[0]
    if verbose:
        print(f'torchquantum {states_tq}')

    assert(torch.allclose(states_tq, qdev.states.full_tensor().cpu()))

    if rank == '0':
        print('test passed!')

if __name__ == "__main__":
    test_dqd()