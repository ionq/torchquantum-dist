import torch

from . import matrices, functional


class GeneralEncoder(torch.nn.Module):
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    def forward(self, q_dev, x):
        for info in self.func_list:
            if callable(matrices.GATE_MAT_DICT[info["func"]]):
                params = x[:, info["input_idx"]]
            else:
                params = None
            getattr(functional, info["func"])(q_dev, info["wires"], params=params)

    def inverse(self, q_dev, x):
        for info in reversed(self.func_list):
            name = info["func"].lower()
            if callable(matrices.GATE_MAT_DICT[name]):
                params = x[:, info["input_idx"]]
            else:
                params = None
            getattr(functional, f"{name}_inv")(q_dev, info["wires"], params=params)


def AmplitudeEncoder(q_dev, amplitudes):
    q_dev.load_amplitudes(amplitudes)
