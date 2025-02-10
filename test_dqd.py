import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from distributed_quantum_device import DistributedQuantumDevice

def test_dqd():
    nq = 3
    dqd = DistributedQuantumDevice(
        nq,
        device=f'cuda',
        world_sz=2,
    )
    
    print(dqd.states.shape)
    print(dqd._states.to_local().shape)
    print(dqd._states.full_tensor().shape)


if __name__ == "__main__":
    test_dqd()