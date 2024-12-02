#%%


import numpy as np

from sas_rmc.array_cache import array_cache
from sas_rmc.polarizer import Polarizer
from sas_rmc.vector import cross
from sas_rmc.particles import FormResult


@array_cache(max_size=5_000)
def sum_array_list(array_list: list[np.ndarray]) -> np.ndarray:
    ARRAY_LIST_BASE_CASE = 8 # This number can result in a recursion error if it's set too low. Beware.
    if len(array_list) < ARRAY_LIST_BASE_CASE:
        return np.sum(array_list, axis = 0)
    divisions = max(2, int(np.sqrt(len(array_list))))
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
    q = (qx, qy, 0)
    q_square = q_squared(qx, qy)
    mqm = cross(q, cross((fm_x, fm_y, fm_z), q))
    return [mq_comp / q_square for mq_comp in mqm]

def box_intensity(form_results: list[FormResult], box_volume: float, qx: np.ndarray, qy: np.ndarray, polarizer: Polarizer, rescale_factor: float = 1) -> np.ndarray:
    fn = nuclear_amplitude(form_results)
    fmx, fmy, fmz = magnetic_amplitude(form_results, qx, qy)
    return 1e8 * rescale_factor * polarizer.intensity_polarization(fn, fmx, fmy, fmz) / box_volume


if __name__ == "__main__":
    print(sum_array_list([np.ones((3,3))]))
        
#%%
        
        