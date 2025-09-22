# adapted from: https://colab.research.google.com/drive/1hxs1_PMJR7CpPm9bTQGoU3P0iFOY6NlO#scrollTo=f9HHc46yRBnJ
import torch

from .matrices import GATE_MAT_DICT
from .encoder import GeneralEncoder
from .functional import InvertiblePostUnitaryStep, gate


class InvertibleUnitary(torch.nn.Module):
    def __init__(self, gates, error_prob=0.0):
        super().__init__()
        self.gates = torch.nn.ModuleList(gates)
        self.error_prob = error_prob
        self.NOISE_GATES = ['x', 'y', 'z', 'i']

    def forward(self, qdev, inp):
        for i in range(len(self.gates)):
            if isinstance(self.gates[i], GeneralEncoder):
                self.gates[i](qdev, inp)
            else:  # it's an `Op`
                self.gates[i](qdev)
                if self.error_prob > 0:
                    rand = torch.rand(1)
                    wires = self.gates[i].wires
                    if len(wires) == 1 and rand < self.error_prob:
                            gate(self.NOISE_GATES[(rand*3/self.error_prob).int()], qdev, wires)
                    elif len(wires) == 2 and rand < self.error_prob * 16/15:
                        randint = (rand * 15 / self.error_prob).int()
                        randix1 = randint % 4
                        randix2 = randint // 4
                        gate(torch.kron(GATE_MAT_DICT[self.NOISE_GATES[randix1]], GATE_MAT_DICT[self.NOISE_GATES[randix2]]), qdev, wires)
                    else:
                        continue
        qdev._states = InvertiblePostUnitaryStep.apply(qdev._states, qdev._invertible_dummy)
