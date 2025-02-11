import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from distributed_quantum_device import DistributedQuantumDevice

def test_dqd():
    rank = os.environ['LOCAL_RANK']
    nq = 3
    world_sz = 2
    dqd = DistributedQuantumDevice(
        nq,
        device=f'cuda',
        world_sz=world_sz
    )

    dqd.x(wires=[0])
    print(f'after {rank} {dqd.lin.weight}')
    if rank == '0':
        print(f'done {dqd.lin.weight}')
        #print(f'done {dqd.lin.weight.full_tensor()}')


if __name__ == "__main__":
    test_dqd()