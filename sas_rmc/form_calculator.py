#%%

from enum import Enum

import numpy as np

from sas_rmc.array_cache import array_cache
from sas_rmc.detector import Polarization
from sas_rmc.vector import cross
from sas_rmc.particles import FormResult


@array_cache(max_size=1_000)
def mod(arr: np.ndarray) -> np.ndarray:
    return np.real(arr * np.conj(arr))


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
    
def nuclear_amplitude(form_results: list[FormResult]) -> np.ndarray:
    return sum_array_list([form_result.form_nuclear for form_result in form_results])
    
@array_cache(max_size=5_000)
def q_squared(qx: np.ndarray, qy: np.ndarray, offset: float = 1e-16) -> np.ndarray:
    qq = qx**2 + qy**2
    return np.where(qq !=0 , qq, offset)

def get_form_magnetic_x(form_result: FormResult) -> np.ndarray:
    return form_result.form_magnetic_x

def get_form_magnetic_y(form_result: FormResult) -> np.ndarray:
    return form_result.form_magnetic_y

def get_form_magnetic_z(form_result: FormResult) -> np.ndarray:
    return form_result.form_magnetic_z

def magnetic_amplitude(form_results: list[FormResult], qx: np.ndarray, qy: np.ndarray) -> list[np.ndarray]:
    fm_x, fm_y, fm_z = [sum_array_list([getter_func(form_result) for form_result in form_results]) for getter_func in (get_form_magnetic_x, get_form_magnetic_y, get_form_magnetic_z)]
    q = [qx, qy, 0]
    q_square = q_squared(qx, qy)
    mqm = cross(q, cross([fm_x, fm_y, fm_z], q))
    return [mq_comp / q_square for mq_comp in mqm]

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

def box_intensity(form_results: list[FormResult], box_volume: float, qx: np.ndarray, qy: np.ndarray, rescale_factor: float = 1, polarization: Polarization = Polarization.UNPOLARIZED, field_direction: FieldDirection = FieldDirection.Y) -> np.ndarray:
    fn = nuclear_amplitude(form_results)
    fmx, fmy, fmz = magnetic_amplitude(form_results, qx, qy)
    return 1e8 * rescale_factor * intensity_polarization(fn, fmx, fmy, fmz, polarization, field_direction=field_direction) / box_volume





if __name__ == "__main__":
    print(sum_array_list([np.ones(3,3)]))
        
#%%
        
        