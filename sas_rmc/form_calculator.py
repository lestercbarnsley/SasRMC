#%%

from enum import Enum
from typing import Callable, List

import numpy as np

from .array_cache import array_cache
from .box_simulation import Box
from .result_calculator import FormResult, ResultCalculator
from .detector import Polarization
from .vector import cross
from . import constants

PI = constants.PI

mod = lambda arr: np.real(arr * np.conj(arr)) # this function is NOT a good candidate for caching

class FieldDirection(Enum):
    X = "X"
    Y = "Y"
    Z = "Z"

@array_cache(max_size=5000)
def _sum_array_list_low_lvl(array_list: List[np.ndarray]) -> np.ndarray:
    return np.sum(array_list, axis = 0) # This is defined so we can use the cache

@array_cache(max_size=5_000)
def sum_array_list(array_list: List[np.ndarray]) -> np.ndarray:
    divisions = int(np.sqrt(len(array_list)))
    partial_sums = [_sum_array_list_low_lvl(array_list[i::divisions]) for i in range(divisions)]
    return _sum_array_list_low_lvl(partial_sums)#sum_array_list(array_list)

def form_result_adder(form_results: List[FormResult], getter_fn: Callable[[FormResult], np.ndarray], rescale: float = 1) -> np.ndarray:
    array_list = [getter_fn(form_result) for form_result in form_results]
    return np.sqrt(rescale) * sum_array_list(array_list)
    
def nuclear_amplitude(form_results: List[FormResult], rescale_factor: float = 1) -> np.ndarray:
    form_nuclear_getter =  lambda form_result: form_result.form_nuclear
    return form_result_adder(form_results, form_nuclear_getter, rescale=rescale_factor)

@array_cache(max_size=5_000)
def q_squared(qx: np.ndarray, qy: np.ndarray, offset: float = 1e-16) -> np.ndarray:
    qq = qx**2 + qy**2
    return np.where(qq !=0 , qq, offset)

def magnetic_amplitude(form_results: List[FormResult], qx: np.ndarray, qy: np.ndarray, magnetic_rescale:float = 1) -> List[np.ndarray]:
    getters = [
        lambda form_result: form_result.form_magnetic_x, 
        lambda form_result: form_result.form_magnetic_y, 
        lambda form_result: form_result.form_magnetic_z 
        ]
    fm_x, fm_y, fm_z = [form_result_adder(form_results, getter_fn, magnetic_rescale) for getter_fn in getters]
    q = [qx, qy, 0]
    q_square = q_squared(qx, qy)
    mqm = cross(q, cross([fm_x, fm_y, fm_z], q))
    return [mq_comp / q_square for mq_comp in mqm]

def form_polarization_x(fn: np.ndarray, fmx: np.ndarray, fmy: np.ndarray, fmz: np.ndarray):
    minus_minus_fn = lambda : fn + fmx
    plus_plus_fn = lambda : fn - fmx
    minus_plus_fn = lambda : -fmy - 1j * fmz
    plus_minus_fn = lambda : -fmy + 1j * fmz
    return minus_minus_fn, plus_plus_fn, minus_plus_fn, plus_minus_fn

def form_polarization_y(fn: np.ndarray, fmx: np.ndarray, fmy: np.ndarray, fmz: np.ndarray):
    minus_minus_fn = lambda : fn + fmy
    plus_plus_fn = lambda : fn - fmy
    minus_plus_fn = lambda : -fmx - 1j * fmz
    plus_minus_fn = lambda : -fmx + 1j * fmz
    return minus_minus_fn, plus_plus_fn, minus_plus_fn, plus_minus_fn

def form_polarization_z(fn: np.ndarray, fmx: np.ndarray, fmy: np.ndarray, fmz: np.ndarray):
    # Is this really correct?
    minus_minus_fn = lambda : fn + fmz
    plus_plus_fn = lambda : fn - fmz
    minus_plus_fn = lambda : -fmx - 1j * fmy
    plus_minus_fn = lambda : -fmx + 1j * fmy
    return minus_minus_fn, plus_plus_fn, minus_plus_fn, plus_minus_fn

def intensity_polarization(fn: np.ndarray, fmx: np.ndarray, fmy: np.ndarray, fmz: np.ndarray, polarization: Polarization, field_direction: FieldDirection = FieldDirection.Y) -> np.ndarray:
    form_polarization = {
        FieldDirection.X : form_polarization_x,
        FieldDirection.Y : form_polarization_y,
        FieldDirection.Z: form_polarization_z
    }[field_direction]
    minus_minus_fn, plus_plus_fn, minus_plus_fn, plus_minus_fn = form_polarization(fn, fmx, fmy, fmz)
    polarization_splitter = {
        Polarization.MINUS_MINUS: [minus_minus_fn],
        Polarization.PLUS_PLUS: [plus_plus_fn],
        Polarization.MINUS_PLUS: [minus_plus_fn],
        Polarization.PLUS_MINUS: [plus_minus_fn],
        Polarization.SPIN_UP: [minus_minus_fn,minus_plus_fn], 
        Polarization.SPIN_DOWN: [plus_plus_fn,plus_minus_fn],
        Polarization.UNPOLARIZED: [minus_minus_fn,minus_plus_fn,plus_plus_fn,plus_minus_fn]
    }
    polarization_functions = polarization_splitter[polarization]
    return np.sum([mod(polarization_fn()) for polarization_fn in polarization_functions], axis=0) / (2 if polarization == Polarization.UNPOLARIZED else 1)

def box_intensity(form_results: List[FormResult], box_volume: float, qx: np.ndarray, qy: np.ndarray, rescale_factor: float = 1, magnetic_rescale: float = 1, polarization: Polarization = Polarization.UNPOLARIZED, field_direction: FieldDirection = FieldDirection.Y) -> np.ndarray:
    fn = nuclear_amplitude(form_results, rescale_factor=rescale_factor)
    fmx, fmy, fmz = magnetic_amplitude(form_results, qx, qy, magnetic_rescale=magnetic_rescale)
    return 1e8 * intensity_polarization(fn, fmx, fmy, fmz, polarization, field_direction=field_direction) / box_volume

def box_intensity_average(box_list: List[Box], result_calculator: ResultCalculator, rescale_factor: float = 1, magnetic_rescale: float = 1, polarization: Polarization = Polarization.UNPOLARIZED, field_direction: FieldDirection = FieldDirection.Y) -> np.ndarray:
    return np.average(
        [box_intensity(
            [result_calculator.form_result(p) for p in box.particles],
            box.volume, result_calculator.qx_array, result_calculator.qy_array,
            rescale_factor=rescale_factor, 
            magnetic_rescale=magnetic_rescale, 
            polarization=polarization,
            field_direction=field_direction) for box in box_list],
        axis = 0
    )




if __name__ == "__main__":
    pass
        
#%%
        
        