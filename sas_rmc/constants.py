#%%

from collections.abc import Callable
import functools
from typing import ParamSpec, Type, TypeVar

import numpy as np
from scipy import constants as scipy_constants

def get_physical_constant(constant_name: str) -> float:
    return scipy_constants.physical_constants[constant_name][0]

PI = np.pi
GAMMA_N = abs(get_physical_constant('neutron mag. mom. to nuclear magneton ratio')) # This value is unitless
R_0 = get_physical_constant('classical electron radius')
BOHR_MAG = get_physical_constant('Bohr magneton')
B_H_IN_INVERSE_AMP_METRES = (GAMMA_N * R_0 / 2) / BOHR_MAG

RNG = np.random.default_rng()

def non_zero_list(ls: list) -> bool:
    return bool(np.sum(np.array(ls)**2))

# string names
NUCLEAR_RESCALE = "Nuclear rescale"
MAGNETIC_RESCALE = "Magnetic rescale"

T = TypeVar('T')

def validate_fields(cls: Type[T], data: dict) -> T:
    if hasattr(cls, '__iter__'):
        return cls(validate_fields(cls.__args__[0], d) for d in data)
    if not hasattr(cls, '__dataclass_fields__'):
        return cls(data)
    if not isinstance(data, dict):
        return validate_fields(cls, data.__dict__)
    d = {k : validate_fields(v.type, data[k])
            for k, v in cls.__dataclass_fields__.items()
            if k in data}
    return cls(**d)

P = ParamSpec('P')
R = TypeVar('R')

def validate_decorator(func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        res = func(*args, **kwargs)
        return validate_fields(type(res), res.__dict__)
    return wrapper


