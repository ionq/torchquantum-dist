import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.distributed.tensor import distribute_tensor, Replicate, Shard

import distributed_quantum_device as dqd
from distributed_quantum_device import DistributedQuantumDevice

def test_dqd():
    rank = os.environ['LOCAL_RANK']
    nq = 3
    world_sz = 2
    qdev = DistributedQuantumDevice(
        nq,
        device=f'cuda',
        world_sz=world_sz
    )
    #print(qdev.states @ torch.ones((2,2), dtype=torch.float, device=f'cuda:{rank}'))
    #print(torch.einsum('...ij,...jk->...ik', qdev.states, torch.ones((2,2), device=f'cuda:{rank}')))
    #print(qdev.states @ distribute_tensor(torch.ones((2,2), dtype=torch.float), device_mesh=qdev.device_mesh, placements=[Replicate()]))
    #print(torch.einsum('...ij,...jk->...ik', qdev.states, distribute_tensor(torch.ones((2,2), dtype=torch.float), device_mesh=dqd.device_mesh, placements=[Replicate()])))
    qdev.y(wires=[2])

    x_gate = dqd.X(wires=[0])
    x_gate(qdev)
    
    qdev.rz(wires=[0], params=torch.ones(1)*torch.pi/3)
    
    print(f'after {rank} {qdev.states}')
    if rank == '0':
        #print(f'done {qdev.states}')
        print(f'done {qdev.states.full_tensor()}')
        pass


if __name__ == "__main__":
    test_dqd()