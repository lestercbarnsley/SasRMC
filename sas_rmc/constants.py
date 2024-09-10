#%%


from typing import ParamSpec, TypeVar

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


