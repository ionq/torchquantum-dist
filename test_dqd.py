import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from torch.distributed.device_mesh import init_device_mesh
from distributed_quantum_device import DistributedQuantumDevice

def test_dqd():
    nq = 32
    device_mesh = init_device_mesh(
        'gpu',
        (2,)
    )
    placements = {}
    DistributedQuantumDevice(
        nq,
        device="gpu",
        device_mesh=device_mesh,
        placements=placements
    )