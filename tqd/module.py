# adapted from: https://colab.research.google.com/drive/1hxs1_PMJR7CpPm9bTQGoU3P0iFOY6NlO#scrollTo=f9HHc46yRBnJ
import torch
from typing import Union

from .matrices import GATE_MAT_DICT
from .encoder import GeneralEncoder
from .functional import InvertiblePostUnitaryStep, gate


class InvertibleUnitary(torch.nn.Module):
    def __init__(self, gates, error_probs: Union[list[float], float] = 0.0):
        super().__init__()
        self.gates = torch.nn.ModuleList(gates)
        if isinstance(error_probs, float):
            error_probs = 2 * [error_probs]
        self.error_1q = error_probs[0]
        self.error_2q = error_probs[1]
        self.NOISE_GATES = ["x", "y", "z", "i"]

    def forward(self, qdev, inp):
        for i in range(len(self.gates)):
            if isinstance(self.gates[i], GeneralEncoder):
                self.gates[i](qdev, inp)
            else:  # it's an `Op`
                self.gates[i](qdev)
                rand = torch.rand(1)
                wires = self.gates[i].wires
                if len(wires) == 1 and self.error_1q > 0 and rand < self.error_1q:
                    gate(
                        self.NOISE_GATES[(rand * 3 / self.error_1q).int()], qdev, wires
                    )
                elif (
                    len(wires) == 2
                    and self.error_2q > 0
                    and rand < self.error_2q * 16 / 15
                ):
                    randint = (rand * 15 / self.error_2q).int()
                    randix1 = randint % 4
                    randix2 = randint // 4
                    gate(
                        torch.kron(
                            GATE_MAT_DICT[self.NOISE_GATES[randix1]],
                            GATE_MAT_DICT[self.NOISE_GATES[randix2]],
                        ),
                        qdev,
                        wires,
                    )
                else:
                    continue
        qdev._states = InvertiblePostUnitaryStep.apply(
            qdev._states, qdev._invertible_dummy
        )
