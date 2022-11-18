#%%
from typing import List

import numpy as np

from .box_simulation import Box
from .result_calculator import NumericalProfileCalculator
from .form_calculator import mod
from . import constants, Vector

PI = constants.PI

def structure_factor(q_array: np.ndarray, particle_position: Vector, box_position_list: List[Vector]) -> np.ndarray:
    structure_factor_func = lambda pos_2 : np.sinc(q_array * (particle_position - pos_2).mag / PI)
    return np.sum([structure_factor_func(position) for position in box_position_list], axis = 0)

def form_array(box: Box, profile_calculator: NumericalProfileCalculator)-> np.ndarray:
    box_position_list = [particle.position for particle in box.particles]
    q_array = profile_calculator.q_array
    form = lambda p : profile_calculator.form_profile(p) * structure_factor(q_array, p.position, box_position_list)
    return np.sum([form(particle) for particle in box.particles], axis = 0)

def box_profile_calculator(box: Box, profile_calculator: NumericalProfileCalculator):
    total_form = form_array(box, profile_calculator)
    return mod(total_form) / box.volume

if __name__ == "__main__":
    pass


#%%
