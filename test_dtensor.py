import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.distributed import get_world_size, init_process_group, get_node_local_rank
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
from distributed_quantum_device import DistributedQuantumDevice

import torch
from torch.distributed.tensor import DTensor, Shard, Replicate, distribute_tensor, distribute_module, init_device_mesh

# construct a device mesh with available devices (multi-host or single host)
device_mesh = init_device_mesh("cuda", (2,))
# if we want to do channel-wise sharding
channel_placement=[Shard(2)]

big_tensor = torch.randn(888, 12, 6)
# distributed tensor returned will be sharded across the dimension specified in placements
rowwise_tensor = distribute_tensor(big_tensor, device_mesh=device_mesh, placements=channel_placement)
print(rowwise_tensor.to_local().shape)
print(rowwise_tensor.full_tensor().shape)
