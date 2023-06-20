#%%
from typing import List
from dataclasses import dataclass, field

import numpy as np

#from .detector import SimulatedDetectorImage
from .box_simulation import Box
from .result_calculator import NumericalProfileCalculator, ProfileCalculatorAnalytical
from .form_calculator import mod
from .scattering_simulation import SimulationParams
from .array_cache import method_array_cache, array_cache
from . import constants, Vector

PI = constants.PI

@array_cache(max_size=5_000)
def array_list_sum(arr_list: List[np.ndarray], bottom_level = False):
    if bottom_level:
        return np.sum(arr_list, axis = 0)
    divs = int(np.sqrt(len(arr_list)))
    return np.sum([array_list_sum(arr_list[i::divs], bottom_level = True)  for i in range(divs) ], axis = 0)

@array_cache(max_size = 5_000)
def structure_factor_func(q_array: np.ndarray, distance: float) -> np.ndarray:# position_1: Vector, position_2: Vector):
    qr = q_array * distance
    #return np.where(qr == 0, 1, np.sin(qr) / qr)
    return np.sinc(qr / PI)

@array_cache(max_size=5_000)
def structure_factor(q_array: np.ndarray, particle_position: Vector, box_position_list: List[Vector]) -> np.ndarray:
    return array_list_sum([structure_factor_func(q_array, particle_position.distance_from_vector(position)) for position in box_position_list])

def form_array(box: Box, profile_calculator: NumericalProfileCalculator)-> np.ndarray:
    box_position_list = [particle.position for particle in box.particles]
    q_array = profile_calculator.q_array
    form = lambda particle, position : profile_calculator.form_profile(particle) * structure_factor(q_array, position, box_position_list)
    return array_list_sum([form(particle, particle.position) for particle in box.particles])

def box_profile_calculator(box: Box, profile_calculator: NumericalProfileCalculator) -> np.ndarray:
    total_form = form_array(box, profile_calculator)
    return mod(total_form) / box.volume



@dataclass
class ProfileFitter:

    box_list: List[Box]
    single_profile_calculator: ProfileCalculatorAnalytical
    experimental_intensity: np.ndarray
    intensity_uncertainty: np.ndarray = field(default_factory=lambda : np.zeros(1000))

    @method_array_cache
    def experimental_uncertainty(self) -> np.ndarray:
        if np.sum(self.intensity_uncertainty**2):
            return self.intensity_uncertainty
        return np.sqrt(self.experimental_intensity)

    def simulated_intensity(self, rescale: float = 1.0) -> np.ndarray:
        return rescale * np.average([self.single_profile_calculator.box_intensity(box) for box in self.box_list], axis= 0)
        #return rescale * np.sum([box_profile_calculator(box, self.single_profile_calculator) for box in self.box_list], axis = 0)

    def fit(self, simulation_params: SimulationParams) -> float:
        rescale = simulation_params.get_value(key = constants.NUCLEAR_RESCALE)
        simulated_intensity = self.simulated_intensity(rescale)
        intensity_uncertainty = self.experimental_uncertainty()
        return np.average((simulated_intensity - self.experimental_intensity)**2 / (intensity_uncertainty**2))
        
        

if __name__ == "__main__":
    pass


#%%
