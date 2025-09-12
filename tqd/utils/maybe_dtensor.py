from typing import Union, Optional

import torch
from torch.distributed.tensor import DTensor, DeviceMesh, Shard


# Small helper subroutines for performing basic DTensor operations if necessary, leaves tensors alone otherwise 

def maybe_to_local(tensor: Union[torch.Tensor, DTensor]) -> Union[torch.Tensor, DTensor]:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor

def maybe_get_dtensor_info(tensor: Union[torch.Tensor, DTensor]) -> (Optional[DeviceMesh], Optional[tuple[Shard]]):
    return (tensor.device_mesh, tensor.placements) if isinstance(tensor, DTensor) else (None, None)

def maybe_from_local(tensor: Union[torch.Tensor, DTensor], device_mesh: Optional[DeviceMesh] = None, placements: Optional[tuple[Shard]] = None) -> Union[torch.Tensor, DTensor]:
    return DTensor.from_local(tensor, device_mesh=device_mesh, placements=placements) if device_mesh and placements else tensor

