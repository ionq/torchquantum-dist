
from . import custom, functional, operator
from .measure import measure_allZ
from .device import DistributedQuantumDevice

for name_ in functional.FUNC_NAMES:
    vars()[name_] = getattr(functional, name_)
    vars()[name_.upper()] = getattr(operator, name_.upper())