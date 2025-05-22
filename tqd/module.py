# adapted from: https://colab.research.google.com/drive/1hxs1_PMJR7CpPm9bTQGoU3P0iFOY6NlO#scrollTo=f9HHc46yRBnJ
import torch

from .encoder import GeneralEncoder
from .functional import InvertiblePostUnitaryStep


class InvertibleUnitary(torch.nn.Module):
    def __init__(self, gates):
        super().__init__()
        self.gates = torch.nn.ModuleList(gates)

    def forward(self, qdev, inp):
        for i in range(len(self.gates)):
            if isinstance(self.gates[i], GeneralEncoder):
                self.gates[i](qdev, inp)
            else:  # it's an `Op`
                self.gates[i](qdev)
            if inp.device.index == 0:
                print(f'  after {type(self.gates[i]).__name__}, qdev._states: {qdev._states}\n    qdev._wire_order: {qdev._wire_order}\n   qdev.states: {qdev.states}')
        qdev._states = InvertiblePostUnitaryStep.apply(qdev._states, qdev._invertible_dummy)