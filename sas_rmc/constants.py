#%%

import numpy as np
from scipy import constants as sp_constants

physical_constants = sp_constants.physical_constants

def get_physical_constant(constant_str: str) -> float:
    val, *_ =  physical_constants[constant_str]
    return val

PI = np.pi
GAMMA_N = np.abs(get_physical_constant('neutron mag. mom. to nuclear magneton ratio')) # This value is unitless
R_0 = get_physical_constant('classical electron radius')
BOHR_MAG = get_physical_constant('Bohr magneton')
B_H_IN_INVERSE_AMP_METRES = (GAMMA_N * R_0 / 2) / BOHR_MAG

RNG = np.random.default_rng()

#NON_ZERO_LIST = lambda ls: np.sum((np.array(ls)**2))

def non_zero_list(ls: list) -> float:
    return np.sum(np.array(ls)**2)

# string names
NUCLEAR_RESCALE = "Nuclear rescale"
MAGNETIC_RESCALE = "Magnetic rescale"

