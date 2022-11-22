#%%
from typing import List
from dataclasses import dataclass, field

import numpy as np

#from .detector import SimulatedDetectorImage
from .box_simulation import Box
from .result_calculator import NumericalProfileCalculator
from .form_calculator import mod
from .scattering_simulation import SimulationParams
from .array_cache import method_array_cache
from . import constants, Vector

PI = constants.PI

def structure_factor(q_array: np.ndarray, particle_position: Vector, box_position_list: List[Vector]) -> np.ndarray:
    structure_factor_func = lambda pos_2 : np.sinc(q_array * (particle_position - pos_2).mag / PI)
    return np.sum([structure_factor_func(position) for position in box_position_list], axis = 0)

def form_array(box: Box, profile_calculator: NumericalProfileCalculator)-> np.ndarray:
    box_position_list = [particle.position for particle in box.particles]
    q_array = profile_calculator.q_array
    form = lambda particle, position : profile_calculator.form_profile(particle) * structure_factor(q_array, position, box_position_list)
    return np.sum([form(particle, particle.position) for particle in box.particles], axis = 0)

def box_profile_calculator(box: Box, profile_calculator: NumericalProfileCalculator) -> np.ndarray:
    total_form = form_array(box, profile_calculator)
    return mod(total_form) / box.volume


@dataclass
class ProfileFitter:

    box_list: List[Box]
    single_profile_calculator: NumericalProfileCalculator
    experimental_intensity: np.ndarray
    intensity_uncertainty: np.ndarray = field(default_factory=lambda : np.zeros(1000))

    @method_array_cache
    def experimental_uncertainty(self) -> np.ndarray:
        if np.sum(self.intensity_uncertainty**2):
            return self.intensity_uncertainty
        return np.sqrt(self.experimental_intensity)

    def fit(self, simulation_params: SimulationParams) -> float:
        rescale = simulation_params.get_value(key = constants.NUCLEAR_RESCALE)
        simulated_intensity = rescale * np.sum([box_profile_calculator(box, self.single_profile_calculator) for box in self.box_list], axis = 0)
        intensity_uncertainty = self.experimental_uncertainty()
        return np.average((simulated_intensity - self.experimental_intensity)**2 / (intensity_uncertainty**2))
        
        

if __name__ == "__main__":
    pass


#%%
