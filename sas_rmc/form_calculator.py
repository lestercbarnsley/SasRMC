#%%

from typing import Callable, List

import numpy as np

from .box_simulation import Box
from .particle import FormResult
from .detector import Polarization
from .vector import cross

PI = np.pi

mod = lambda arr: np.real(arr * np.conj(arr))

form_result_adder = lambda form_results, getter_fn, rescale = 1: np.sqrt(rescale) * np.sum([getter_fn(form_result) for form_result in form_results], axis = 0)

def form_result_adder_faster(form_results: List[FormResult], getter_fn: Callable[[FormResult], np.ndarray], rescale: float = 1) -> np.ndarray:
    shape = getter_fn(form_results[0]).shape
    arr = np.zeros((len(form_results), shape[0], shape[1]))
    for i, form_result in enumerate(form_results):
        arr[i,:,:] = getter_fn(form_result)
    return np.sqrt(rescale) * np.sum(arr, axis = 0)
    # Mark this for deletion. It might be faster, but it's a less elegant piece of code


def nuclear_amplitude(form_results: List[FormResult], rescale_factor = 1):
    form_nuclear_getter =  lambda form_result: form_result.form_nuclear
    return form_result_adder(form_results, form_nuclear_getter, rescale=rescale_factor)

def magnetic_amplitude(form_results: List[FormResult], qx, qy, magnetic_rescale = 1):
    getters = [
        lambda form_result: form_result.form_magnetic_x, 
        lambda form_result: form_result.form_magnetic_y, 
        lambda form_result: form_result.form_magnetic_z 
        ]
    fm_x, fm_y, fm_z = [form_result_adder(form_results, getter_fn, magnetic_rescale) for getter_fn in getters]
    q = [qx, qy, 0]
    q_squared = qx**2 + qy**2+ 1e-16 # Avoid a divide by zero warning
    mqm = cross(q, cross([fm_x, fm_y, fm_z], q))
    return [mq_comp / q_squared for mq_comp in mqm]

def intensity_polarization(fn: np.ndarray, fmx: np.ndarray, fmy: np.ndarray, fmz: np.ndarray, polarization: Polarization):
    minus_minus_fn = lambda : fn + fmy
    plus_plus_fn = lambda : fn - fmy
    minus_plus_fn = lambda : -fmx - 1j * fmz
    plus_minus_fn = lambda : -fmx + 1j * fmz # Use lambda here so these summations are only computed as they are needed
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

def box_intensity(box: Box, qx, qy, rescale_factor = 1, magnetic_rescale = 1, polarization: Polarization = Polarization.UNPOLARIZED):
    form_results = [particle.form_result(qx, qy) for particle in box.particles]
    fn = nuclear_amplitude(form_results, rescale_factor=rescale_factor)
    fmx, fmy, fmz = magnetic_amplitude(form_results, qx, qy, magnetic_rescale=magnetic_rescale)
    return 1e8 * intensity_polarization(fn, fmx, fmy, fmz, polarization) / box.volume

def box_intensity_average(box_list: List[Box], qx, qy, rescale_factor = 1, magnetic_rescale = 1, polarization: Polarization = Polarization.UNPOLARIZED):
    return np.average(
        [box_intensity(
            box, qx, qy,
            rescale_factor=rescale_factor, 
            magnetic_rescale=magnetic_rescale, 
            polarization=polarization) for box in box_list],
        axis = 0
    )




if __name__ == "__main__":
    pass
        
#%%
        
        