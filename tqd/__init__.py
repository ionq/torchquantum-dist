
from . import custom, functional, module, operator
from .device import DistributedQuantumDevice
from .encoder import GeneralEncoder
from .measure import measure_allZ

for name_ in functional.GATE_MAT_DICT.keys():
    vars()[name_] = getattr(functional, name_)
    vars()[name_.upper()] = getattr(operator, name_.upper())