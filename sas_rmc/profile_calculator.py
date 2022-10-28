#%%

import numpy as np

from .box_simulation import Box
from .result_calculator import NumericalProfileCalculator
from .form_calculator import mod
from . import constants

PI = constants.PI

def box_profile_calculator(box: Box, profile_calculator: NumericalProfileCalculator):
    total_form = 0
    for p in box.particles:
        form = profile_calculator.form_profile(p)
        position = p.position
        total_structure = 0
        for p2 in box.particles:
            delta_position = (position-p2.position).mag
            structure_factor = np.sinc(profile_calculator.q_array * delta_position / PI)
            total_structure += structure_factor
        total_form += form * total_structure
    return mod(total_form)

if __name__ == "__main__":
    pass


#%%
