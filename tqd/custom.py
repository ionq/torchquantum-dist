from functools import partial

from . import functional, operator

def register_gate(name, mat, has_params):
    globals()[name.lower()] = partial(functional.gate_wrapper, name, mat, "bmm")
    globals()[name.upper()] = operator.OpFactory(name.lower(), globals(), has_params=has_params)