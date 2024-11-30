#%%


import numpy as np
from numpy import typing as npt
from scipy import constants as scipy_constants

def get_physical_constant(constant_name: str) -> float:
    return scipy_constants.physical_constants[constant_name][0]

PI = np.pi
GAMMA_N = abs(get_physical_constant('neutron mag. mom. to nuclear magneton ratio')) # This value is unitless
R_0 = get_physical_constant('classical electron radius')
BOHR_MAG = get_physical_constant('Bohr magneton')
B_H_IN_INVERSE_AMP_METRES = (GAMMA_N * R_0 / 2) / BOHR_MAG

RNG = np.random.default_rng()

def np_max(array: npt.NDArray[np.floating]) -> float:
    return np.max(array).item()

def np_min(array: npt.NDArray[np.floating]) -> float:
    return np.min(array).item()

def np_average(array: list[float], weights: list[float] | None = None) -> float:
    return np.average(array, weights=weights).item()

def np_sum(array: npt.NDArray[np.floating]) -> float:
    return np.sum(array).item()

def np_prod(array: npt.NDArray[np.floating]) -> float:
    return np.prod(array).item()



# string names
NUCLEAR_RESCALE = "Nuclear rescale"
MAGNETIC_RESCALE = "Magnetic rescale"


