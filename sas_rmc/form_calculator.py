#%%

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator

import numpy as np

from sas_rmc.array_cache import array_cache
from sas_rmc.box_simulation import Box
from sas_rmc.detector import Polarization
from sas_rmc.vector import cross
from sas_rmc.particles import FormResult


@array_cache(max_size=1_000)
def mod(arr: np.ndarray) -> np.ndarray:
    return np.real(arr * np.conj(arr))
# why did I think this?

class FieldDirection(Enum):
    X = "X"
    Y = "Y"
    Z = "Z"


@array_cache(max_size=5_000)
def sum_array_list(array_list: list[np.ndarray]) -> np.ndarray:
    if len(array_list) < 3:
        return np.sum(array_list, axis = 0)
    divisions = 2
    return sum_array_list([sum_array_list(array_list[i::divisions]) for i in range(divisions)])

@array_cache(max_size = 5000)
def add_form_results(form_results: list[FormResult]) -> FormResult:
    return FormResult(
        form_nuclear=sum_array_list([form_result.form_nuclear for form_result in form_results]),
        form_magnetic_x=sum_array_list([form_result.form_magnetic_x for form_result in form_results]),
        form_magnetic_y=sum_array_list([form_result.form_magnetic_y for form_result in form_results]),
        form_magnetic_z=sum_array_list([form_result.form_magnetic_z for form_result in form_results])
    )
    
def nuclear_amplitude(form_results: list[FormResult]) -> np.ndarray:
    return add_form_results(form_results).form_nuclear
    
@array_cache(max_size=5_000)
def q_squared(qx: np.ndarray, qy: np.ndarray, offset: float = 1e-16) -> np.ndarray:
    qq = qx**2 + qy**2
    return np.where(qq !=0 , qq, offset)

def magnetic_amplitude(form_results: list[FormResult], qx: np.ndarray, qy: np.ndarray) -> list[np.ndarray]:
    added_form_results = add_form_results(form_results)
    fm_x, fm_y, fm_z = added_form_results.form_magnetic_x, added_form_results.form_magnetic_y, added_form_results.form_magnetic_z
    q = [qx, qy, 0]
    q_square = q_squared(qx, qy)
    mqm = cross(q, cross([fm_x, fm_y, fm_z], q))
    return [mq_comp / q_square for mq_comp in mqm]
    

def form_polarization(fn: np.ndarray, fm_para: np.ndarray, fm_perp_1: np.ndarray, fm_perp_2: np.ndarray):
    def minus_minus() -> np.ndarray:
        return fn + fm_para
    def plus_plus() -> np.ndarray:
        return fn - fm_para
    def minus_plus() -> np.ndarray:
        return -fm_perp_1 - 1j * fm_perp_2
    def plus_minus() -> np.ndarray:
        return -fm_perp_1 + 1j * fm_perp_2
    return minus_minus, plus_plus, minus_plus, plus_minus
    '''
    minus_minus_fn = lambda : fn + fmx
    plus_plus_fn = lambda : fn - fmx
    minus_plus_fn = lambda : -fmy - 1j * fmz
    plus_minus_fn = lambda : -fmy + 1j * fmz
    return minus_minus_fn, plus_plus_fn, minus_plus_fn, plus_minus_fn'''

def form_polarization_x(fn: np.ndarray, fmx: np.ndarray, fmy: np.ndarray, fmz: np.ndarray):
    return form_polarization(fn, fm_para = fmx, fm_perp_1=fmy, fm_perp_2=fmz)

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

def minus_minus(fn: np.ndarray, fm_para: np.ndarray, fm_perp_1: np.ndarray, fm_perp_2: np.ndarray) -> np.ndarray:
    return mod(fn + fm_para)

def plus_plus(fn: np.ndarray, fm_para: np.ndarray, fm_perp_1: np.ndarray, fm_perp_2: np.ndarray) -> np.ndarray:
    return mod(fn - fm_para)

def minus_plus(fn: np.ndarray, fm_para: np.ndarray, fm_perp_1: np.ndarray, fm_perp_2: np.ndarray) -> np.ndarray:
    return mod(-fm_perp_1 - 1j * fm_perp_2)

def plus_minus(fn: np.ndarray, fm_para: np.ndarray, fm_perp_1: np.ndarray, fm_perp_2: np.ndarray) -> np.ndarray:
    return mod(-fm_perp_1 + 1j * fm_perp_2)

def spin_up(fn: np.ndarray, fm_para: np.ndarray, fm_perp_1: np.ndarray, fm_perp_2: np.ndarray) -> np.ndarray:
    return sum_array_list([pol_func(fn, fm_para, fm_perp_1, fm_perp_2) for pol_func in (minus_minus, minus_plus)])

def spin_down(fn: np.ndarray, fm_para: np.ndarray, fm_perp_1: np.ndarray, fm_perp_2: np.ndarray) -> np.ndarray:
    return sum_array_list([pol_func(fn, fm_para, fm_perp_1, fm_perp_2) for pol_func in (minus_minus, minus_plus)])

def unpolarized(fn: np.ndarray, fm_para: np.ndarray, fm_perp_1: np.ndarray, fm_perp_2: np.ndarray) -> np.ndarray:
    return sum_array_list([pol_func(fn, fm_para, fm_perp_1, fm_perp_2) for pol_func in (spin_up, spin_down)]) / 2 # The two is necessary to normalize unpolarized

def intensity_polarization(fn: np.ndarray, fmx: np.ndarray, fmy: np.ndarray, fmz: np.ndarray, polarization: Polarization, field_direction: FieldDirection = FieldDirection.Y) -> np.ndarray:
    form_set = {
        FieldDirection.X : (fn, fmx, fmy, fmz),
        FieldDirection.Y : (fn, fmy, fmx, fmz),
        FieldDirection.Z: (fn, fmz, fmx, fmy)
    }[field_direction]
    polarization_function = {
        Polarization.MINUS_MINUS: minus_minus,
        Polarization.PLUS_PLUS: plus_plus,
        Polarization.MINUS_PLUS: minus_plus,
        Polarization.PLUS_MINUS: plus_minus,
        Polarization.SPIN_UP: spin_up, 
        Polarization.SPIN_DOWN: spin_down,
        Polarization.UNPOLARIZED: unpolarized
    }[polarization]
    return polarization_function(fn = form_set[0], fm_para=form_set[1], fm_perp_1=form_set[2], fm_perp_2=form_set[3])

def box_intensity(form_results: list[FormResult], box_volume: float, qx: np.ndarray, qy: np.ndarray, rescale_factor: float = 1, magnetic_rescale: float = 1, polarization: Polarization = Polarization.UNPOLARIZED, field_direction: FieldDirection = FieldDirection.Y) -> np.ndarray:
    fn = nuclear_amplitude(form_results, rescale_factor=rescale_factor)
    fmx, fmy, fmz = magnetic_amplitude(form_results, qx, qy, magnetic_rescale=magnetic_rescale)
    return 1e8 * intensity_polarization(fn, fmx, fmy, fmz, polarization, field_direction=field_direction) / box_volume

def box_intensity_average(box_list: list[Box], result_calculator: ResultCalculator, rescale_factor: float = 1, magnetic_rescale: float = 1, polarization: Polarization = Polarization.UNPOLARIZED, field_direction: FieldDirection = FieldDirection.Y) -> np.ndarray:
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
    print(sum_array_list([np.ones(3,3)]))
        
#%%
        
        