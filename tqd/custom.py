from functools import partial

from . import functional, matrices, operator

def register_gate(name, mat):
    name = name.lower()
    if name.lower() not in matrices.GATE_MAT_DICT:
        matrices.GATE_MAT_DICT.update({name: mat})
        globals()[name] = partial(functional.gate, name)
        globals()[f"{name}_inv"] = partial(functional.gate, name, inverse=True)
        globals()[name.upper()] = operator.OpFactory(name.lower(), globals())