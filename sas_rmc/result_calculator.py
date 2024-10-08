#%%
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from sas_rmc.array_cache import method_array_cache
from sas_rmc.detector import Polarization
from sas_rmc.form_calculator import box_intensity, FieldDirection
from sas_rmc.particles.particle import FormResult, Particle
from sas_rmc.scattering_simulation import ScatteringSimulation
from sas_rmc import constants, vector

PI = constants.PI


@dataclass
class ResultCalculator(ABC):
    
    @abstractmethod
    def intensity_result(self, scattering_simulation: ScatteringSimulation) -> np.ndarray:
        pass


def modulated_form_array(particle: Particle, qx_array: np.ndarray, qy_array: np.ndarray) -> FormResult:
    form_result = particle.form_result(qx_array, qy_array)
    position = particle.get_position()
    modulation = np.exp(1j * vector.dot(position.to_tuple(), (qx_array, qy_array)))
    return FormResult(
        form_nuclear=form_result.form_nuclear * modulation,
        form_magnetic_x=form_result.form_magnetic_x * modulation,
        form_magnetic_y=form_result.form_magnetic_y * modulation,
        form_magnetic_z=form_result.form_magnetic_z * modulation
    )


@dataclass
class AnalyticalCalculator(ResultCalculator):
    qx_array: np.ndarray
    qy_array: np.ndarray
    polarization: Polarization
    field_direction: FieldDirection = FieldDirection.Y

    @method_array_cache(cache_holder_index=1)
    def modulated_form_array(self, particle: Particle) -> FormResult:
        return modulated_form_array(
            particle=particle,
            qx_array=self.qx_array,
            qy_array=self.qy_array
        )

    def intensity_result(self, scattering_simulation: ScatteringSimulation) -> np.ndarray:
        return np.average(
            [box_intensity(
                form_results=[self.modulated_form_array(particle) for particle in box.particles], 
                box_volume= box.volume, 
                qx=self.qx_array, 
                qy=self.qy_array, 
                rescale_factor=scattering_simulation.scale_factor.value, 
                polarization=self.polarization, 
                field_direction=self.field_direction
                ) for box in scattering_simulation.box_list],
            axis = 0
            )


            

if __name__ == "__main__":
    pass

#%%
