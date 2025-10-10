from typing import Union, Optional

import torch
from torch.distributed.tensor import DTensor, DeviceMesh, Shard, distribute_tensor


# Small helper subroutines for performing basic DTensor operations if necessary, leaves tensors alone otherwise 

def is_dtensor(tensor: Union[torch.Tensor, DTensor]) -> bool:
    return isinstance(tensor, DTensor)

def maybe_to_local(tensor: Union[torch.Tensor, DTensor]) -> Union[torch.Tensor, DTensor]:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor

def maybe_get_dtensor_info(tensor: Union[torch.Tensor, DTensor]) -> (Optional[DeviceMesh], tuple[Shard]):
    return (tensor.device_mesh, tensor.placements) if isinstance(tensor, DTensor) else (None, ())

def maybe_from_local(tensor: Union[torch.Tensor, DTensor], device_mesh: Optional[DeviceMesh] = None, placements: tuple[Shard] = ()) -> Union[torch.Tensor, DTensor]:
    return DTensor.from_local(tensor, device_mesh=device_mesh, placements=placements) if device_mesh and placements else tensor

def maybe_full_tensor(tensor: Union[torch.Tensor, DTensor]) -> torch.Tensor:
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor

def maybe_distribute_tensor(tensor: Union[torch.Tensor, DTensor], device_mesh: Optional[DeviceMesh] = None, placements: tuple[Shard] = ()) -> Union[torch.Tensor, DTensor]:
    return distribute_tensor(tensor, device_mesh=device_mesh, placements=placements) if device_mesh and placements else tensor
