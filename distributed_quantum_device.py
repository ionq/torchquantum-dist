from typing import List, Optional, Union

import torch
import torch.distributed
import torchquantum as tq
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Placement
from torchquantum.macro import C_DTYPE


class DistributedQuantumDevice(tq.QuantumDevice):
    def __init__(
        self,
        n_wires: int,
        device_name: str = "default",
        bsz: int = 1,
        device: Union[torch.device, str] = "cpu",
        record_op: bool = False,
        device_mesh: Optional[DeviceMesh] = None,
        placements: Optional[List[Placement]] = None
    ):
        """A quantum device that contains the quantum state vector.
        Args:
            n_wires: number of qubits
            device_name: name of the quantum device
            bsz: batch size of the quantum state
            device: which classical computing device to use, 'cpu' or 'cuda'
            record_op: whether to record the operations on the quantum device and then
                they can be used to construct a static computation graph
        """
        super().__init__()
        # number of qubits
        # the states are represented in a multi-dimension tensor
        # from left to right: qubit 0 to n
        self.n_wires = n_wires
        self.device_name = device_name
        self.bsz = bsz
        self.device = device
        self.device_mesh = device_mesh
        self.placements = placements

        _state = torch.zeros(2**self.n_wires, dtype=C_DTYPE)
        _state[0] = 1 + 0j  # type: ignore
        _state = torch.reshape(_state, [2] * self.n_wires).to(self.device)
        self.register_buffer("state", _state)

        repeat_times = [bsz] + [1] * len(self.state.shape)  # type: ignore
        self._states = self.state.repeat(*repeat_times)  # type: ignore

        # make this distributed pytorch>=2.5
        self._states = torch.distributed.tensor.DTensor.from_local(
            self._states,
            device_mesh=device_mesh,
            placements=placements,
        )

        self.register_buffer("states", self._states)

        self.record_op = record_op
        self.op_history = []