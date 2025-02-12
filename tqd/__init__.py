from .device import DistributedQuantumDevice
from . import functional
from . import operator

for name_ in functional.FUNC_NAMES:
    vars()[name_] = getattr(functional, name_)
    vars()[name_.upper()] = getattr(operator, name_.upper())