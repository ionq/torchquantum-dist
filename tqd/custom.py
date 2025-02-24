from functools import partial

from . import functional, matrices, operator

def register_gate(name, mat):
    if name.lower() not in matrices.GATE_MAT_DICT:
        matrices.GATE_MAT_DICT.update({name.lower(): mat})
        globals()[name.lower()] = partial(functional.gate, name)
        globals()[name.upper()] = operator.OpFactory(name.lower(), globals())