#%%
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy import special

from sas_rmc import constants, vector
from sas_rmc.box_simulation import Box
from sas_rmc.array_cache import method_array_cache, array_cache
from sas_rmc.detector import Polarization
from sas_rmc.form_calculator import box_intensity, FieldDirection, sum_array_list
from sas_rmc.particles import ParticleArray, FormResult
from sas_rmc.particles.particle_profile import ParticleProfile
from sas_rmc.scattering_simulation import ScatteringSimulation


PI = constants.PI
j0_bessel = special.j0


@dataclass
class ResultCalculator(ABC):
    
    @abstractmethod
    def intensity_result(self, scattering_simulation: ScatteringSimulation) -> np.ndarray:
        pass


def modulated_form_array(particle: ParticleArray, qx_array: np.ndarray, qy_array: np.ndarray) -> FormResult:
    form_result = particle.form_result(qx_array, qy_array)
    position = particle.get_position()
    modulation = np.exp(1j * vector.dot(position.to_tuple(), (qx_array, qy_array)))
    return FormResult(
        form_nuclear=form_result.form_nuclear * modulation,
        form_magnetic_x=form_result.form_magnetic_x * modulation,
        form_magnetic_y=form_result.form_magnetic_y * modulation,
        form_magnetic_z=form_result.form_magnetic_z * modulation
    )

@array_cache
def particle_arrays_from(box: Box) -> list[ParticleArray]:
    return [particle for particle in box.particles if isinstance(particle, ParticleArray)]

@array_cache
def particle_profiles_from(box: Box) -> list[ParticleProfile]:
    return [particle for particle in box.particles if isinstance(particle, ParticleProfile)]


@dataclass
class AnalyticalCalculator(ResultCalculator):
    qx_array: np.ndarray
    qy_array: np.ndarray
    polarization: Polarization
    field_direction: FieldDirection = FieldDirection.Y

    @method_array_cache(cache_holder_index=1)
    def modulated_form_array(self, particle: ParticleArray) -> FormResult:
        return modulated_form_array(
            particle=particle,
            qx_array=self.qx_array,
            qy_array=self.qy_array
        )

    def intensity_result(self, scattering_simulation: ScatteringSimulation) -> np.ndarray:
        return np.average(  
            [box_intensity(
                form_results=[self.modulated_form_array(particle) for particle in particle_arrays_from(box)], 
                box_volume= box.volume, 
                qx=self.qx_array, 
                qy=self.qy_array, 
                rescale_factor=scattering_simulation.scale_factor.value, 
                polarization=self.polarization, 
                field_direction=self.field_direction
                ) for box in scattering_simulation.box_list],
            axis = 0
            )
        # There may be a speed boost associated with caching the box intensity to the box, but this only matters if the total number of boxes exceeds
        # the base case number in the sum array list function
    
    
@array_cache(max_size=40_000)
def structure_factor(q: np.ndarray, distance: float) -> np.ndarray:
    qr = q * distance
    return j0_bessel(qr)


@dataclass
class ProfileCalculator(ResultCalculator):
    q_profile: np.ndarray

    @method_array_cache(cache_holder_index=1)
    def form_profile(self, particle: ParticleProfile) -> np.ndarray:
        return particle.form_profile(self.q_profile)
    
    def structure_factor(self, particle_i: ParticleProfile, particle_j: ParticleProfile) -> np.ndarray:
        distance = particle_i.get_position().distance_from_vector(particle_j.get_position())
        return structure_factor(self.q_profile, distance)
    
    @method_array_cache(cache_holder_index=1, max_size=1000)
    def form_structure_product(self, particle_i: ParticleProfile, particle_j: ParticleProfile) -> np.ndarray:
        return self.form_profile(particle_i) * self.form_profile(particle_j) * self.structure_factor(particle_i, particle_j)

    def box_profile(self, box: Box, rescale_factor: float) -> np.ndarray:
        return (1e8 * rescale_factor / box.volume) * sum_array_list(
            [sum_array_list(
                [self.form_structure_product(particle_i, particle_j) for particle_j in particle_profiles_from(box)]
            ) for particle_i in particle_profiles_from(box)]
        )

    def intensity_result(self, scattering_simulation: ScatteringSimulation) -> np.ndarray:
        return np.average(
            [self.box_profile(
                box, 
                scattering_simulation.scale_factor.value
                ) for box in scattering_simulation.box_list],
            axis = 0
        )
        


            

if __name__ == "__main__":
    pass

#%%
