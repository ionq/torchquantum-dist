import torch

from . import functional, matrices

# Base class for Operator
class Op(torch.nn.Module):
    def __init__(self, func, wires, has_params=True, trainable=True, **unused):
        super().__init__()
        self.func_ = func
        self.wires = wires
        self.has_params = has_params
        self.trainable = trainable
        self.params = None
        if has_params:
            self.params = torch.empty(1)
            if trainable:
                self.params = torch.nn.Parameter(self.params)
    
    def forward(self, qdev, wires=None, params=None):
        self.func_(
            qdev,
            wires if wires is not None else self.wires,
            params=params if params is not None else self.params
        )

# Factory that programattically creates RY from ry, CX from cx, etc
def OpFactory(name, module, has_params=True, trainable=True):
    """
    `name` is lower case
    """
    def __init__(self, wires, **kwargs):
        kwargs.update({'has_params': kwargs.get('has_params', has_params)})
        kwargs.update({'trainable': kwargs.get('trainable', trainable)})
        Op.__init__(self, getattr(module, name), wires, **kwargs)
    newclass = type(name.upper(), (Op, ), {"__init__": __init__})
    return newclass

for name_, mat_ in matrices.GATE_MAT_DICT.items():
    kw = {} if callable(mat_) else {'has_params': False, 'trainable': False}
    vars()[name_.upper()] = OpFactory(name_, functional, **kw)