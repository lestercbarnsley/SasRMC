#%%
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Protocol, Tuple, List

import numpy as np
from scipy.special import j0 as j0_bessel

from .particles import Particle, ParticleComposite
from .particles.particle import magnetic_sld_in_angstrom_minus_2
from .vector import Vector, VectorSpace, broadcast_to_numpy_array, composite_function
from .array_cache import method_array_cache, array_cache
from .box_simulation import Box
from . import constants

PI = constants.PI

@dataclass
class FormResult:
    form_nuclear: np.ndarray
    form_magnetic_x: np.ndarray
    form_magnetic_y: np.ndarray
    form_magnetic_z: np.ndarray


@dataclass
class ResultCalculator(ABC):
    qx_array: np.ndarray
    qy_array: np.ndarray

    @abstractmethod
    def form_result(self, particle: Particle) -> FormResult:
        pass


@dataclass
class AnalyticalCalculator(ResultCalculator):
    
    @method_array_cache(cache_holder_index=1)
    def modulated_form_array_calculator(self, particle: Particle, orientation: Vector) -> Callable[[Vector], np.ndarray]:
        form_array = particle.form_array(self.qx_array, self.qy_array, orientation)
        return lambda position : form_array * np.exp(1j * (position * (self.qx_array, self.qy_array)))

    @method_array_cache(cache_holder_index=1)
    def modulated_form_array(self, particle: Particle, orientation: Vector, position: Vector) -> np.ndarray:
        if isinstance(particle, ParticleComposite):
            return np.sum([self.modulated_form_array(particle_component, particle_component.orientation, particle_component.position) for particle_component in particle.particle_list], axis=0)        
        modulated_array_calculator = self.modulated_form_array_calculator(particle, orientation)
        return modulated_array_calculator(position)

    @method_array_cache(cache_holder_index=1)
    def magnetic_modulated_array_calculator(self, particle: Particle, orientation: Vector, magnetization: Vector) -> Callable[[Vector],Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        magnetic_form_arrays = particle.magnetic_form_array(self.qx_array, self.qy_array, orientation, magnetization)
        return lambda position : [magnetic_form_array * np.exp(1j * (position * (self.qx_array, self.qy_array))) for magnetic_form_array in magnetic_form_arrays]

    @method_array_cache(cache_holder_index=1)
    def magnetic_modulated_array(self, particle: Particle, orientation: Vector, magnetization: Vector, position: Vector) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(particle, ParticleComposite):
            mag_arrs = [self.magnetic_modulated_array(particle_component, particle_component.orientation, particle_component.magnetization, particle_component.position) for particle_component in particle.particle_list]
            get_added_mag_comp = lambda i: np.sum([m[i] for m in mag_arrs], axis=0)
            return [get_added_mag_comp(i) for i in range(3)]
        magnetic_array_calculator = self.magnetic_modulated_array_calculator(particle, orientation, magnetization)
        return magnetic_array_calculator(position)
    
    def form_result(self, particle: Particle) -> FormResult:
        form_nuclear = self.modulated_form_array(particle, particle.orientation, particle.position)
        form_magnetic_x, form_magnetic_y, form_magnetic_z = self.magnetic_modulated_array(particle, particle.orientation, particle.magnetization, particle.position)
        return FormResult(
            form_nuclear=form_nuclear,
            form_magnetic_x=form_magnetic_x,
            form_magnetic_y=form_magnetic_y,
            form_magnetic_z=form_magnetic_z
        )


def numerical_form_array(sld_arr: np.ndarray, vector_space: VectorSpace, qx_array: np.ndarray, qy_array: np.ndarray, xy_axis: int = 2) -> np.ndarray:
    average_projection = lambda arr : np.average(arr, axis = xy_axis)
    flat_sld = np.sum(sld_arr * vector_space.dz, axis = xy_axis)
    x = average_projection(vector_space.x)
    y = average_projection(vector_space.y)
    dx = average_projection(vector_space.dx)
    dy = average_projection(vector_space.dy)
    form_f = lambda qx, qy: np.sum(flat_sld * np.exp(1j * (Vector(qx, qy) * (x, y))) * dx * dy)
    form_calculator = np.frompyfunc(form_f, 2, 1)
    return form_calculator(qx_array, qy_array).astype(np.complex128)

def numerical_form_array_3d(sld_arr: np.ndarray, vector_space: VectorSpace, q_vector: Vector) -> float:
    return np.sum(sld_arr * np.exp(1j * (q_vector * (vector_space.x, vector_space.y, vector_space.z)) * vector_space.dx * vector_space.dy * vector_space.dz))




@dataclass
class ParticleNumerical(Protocol):
    def get_sld(self, relative_position: Vector) -> float:
        pass

    def get_magnetization(self, relative_position: Vector) -> Vector:
        pass

    def delta_sld(self, sld: float) -> float:
        pass


@dataclass
class NumericalCalculator(AnalyticalCalculator):
    vector_space: VectorSpace

    def sld_from_vector_space(self, get_sld: Callable[[Vector], float], delta_sld: Callable[[float], float]) -> np.ndarray:
        element_to_delta_sld = composite_function(
            delta_sld,
            get_sld,
            lambda element: element.position
        )
        return self.vector_space.array_from_elements(element_to_delta_sld)

    @method_array_cache(cache_holder_index=1)
    def modulated_form_array_calculator(self, particle: ParticleNumerical, orientation: Vector) -> Callable[[Vector], np.ndarray]:
        sld_arr = self.sld_from_vector_space(particle.get_sld, particle.delta_sld)
        form_array = numerical_form_array(sld_arr, self.vector_space, self.qx_array, self.qy_array)
        return lambda position : form_array * np.exp(1j * (position * (self.qx_array, self.qy_array)))

    def magnetic_sld_from_vector_space(self, get_magnetization: Callable[[Vector], Vector]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        element_to_magnetization = composite_function(
            get_magnetization,
            lambda element: element.position
        )
        m_field = self.vector_space.field_from_element(element_to_magnetization)
        sld_getters = np.frompyfunc(magnetic_sld_in_angstrom_minus_2, 1, 3)
        mag_sld_x, mag_sld_y, mag_sld_z = sld_getters(m_field)
        return [mag_sld.astype(np.float64) for mag_sld in [mag_sld_x, mag_sld_y, mag_sld_z]]

    @method_array_cache(cache_holder_index=1)
    def magnetic_modulated_array_calculator(self, particle: ParticleNumerical, orientation: Vector, magnetization: Vector) -> Callable[[Vector],Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        magnetic_slds = self.magnetic_sld_from_vector_space(particle.get_magnetization)
        magnetic_form_arrays = [numerical_form_array(magnetic_sld, self.vector_space, self.qx_array, self.qy_array) for magnetic_sld in magnetic_slds]
        return lambda position : [magnetic_form_array * np.exp(1j * (position * (self.qx_array, self.qy_array))) for magnetic_form_array in magnetic_form_arrays]




@dataclass
class NumericalProfileCalculator:
    q_array: np.ndarray
    r_array: np.ndarray
    average_sphere_points: int = 100

    @method_array_cache(cache_holder_index=1)
    def form_profile(self, particle: ParticleNumerical):
        average_sld_fn = lambda r : np.average([particle.delta_sld( particle.get_sld(Vector.random_vector(r))) for _ in range(self.average_sphere_points)])
        average_sld = broadcast_to_numpy_array(self.r_array, average_sld_fn)
        dr = np.gradient(self.r_array)
        sin_q_fn = lambda q : np.sum(average_sld * np.sinc(q * self.r_array / PI) * dr * (self.r_array ** 2)) # Using np.sinc here guarantees sin(qr)/qr = 1 if qr = 0
        return broadcast_to_numpy_array(self.q_array, sin_q_fn)


@dataclass
class ParticleAverageNumerical(Protocol):
    def delta_sld(self, sld: float) -> float:
        pass

    def get_average_sld(self, radius: float) -> float:
        pass


@dataclass
class ProfileCalculator:
    q_array: np.ndarray
    r_array: np.ndarray

    @method_array_cache(cache_holder_index=1)
    def form_profile(self, particle: ParticleAverageNumerical):
        average_sld = broadcast_to_numpy_array(self.r_array, particle.get_average_sld)
        dr = np.gradient(self.r_array)
        sin_q_fn = lambda q : np.sum(average_sld * np.sinc(q * self.r_array / PI) * dr * (self.r_array ** 2)) # Using np.sinc here guarantees sin(qr)/qr = 1 if qr = 0
        return broadcast_to_numpy_array(self.q_array, sin_q_fn)


@array_cache(max_size=40_000)
def structure_factor(q: np.ndarray, distance: float) -> np.ndarray:
    qr = q * distance
    return j0_bessel(qr)


@array_cache(max_size=5_000)
def array_list_sum(arr_list: List[np.ndarray], bottom_level = False):
    if bottom_level:
        return np.sum(arr_list, axis = 0)
    divs = int(np.sqrt(len(arr_list)))
    return np.sum([array_list_sum(arr_list[i::divs], bottom_level = True)  for i in range(divs) ], axis = 0)


@dataclass
class ProfileCalculatorAnalytical:
    q_array: np.ndarray

    @method_array_cache(cache_holder_index=1)
    def form_profile(self, particle: Particle) -> np.ndarray:
        return particle.form_array(self.q_array, 0, orientation=particle.orientation)

    def structure_factor(self, particle_i: Particle, particle_j: Particle) -> np.ndarray:
        distance = particle_i.position.distance_from_vector(particle_j.position)
        return structure_factor(self.q_array, distance)

    @method_array_cache(cache_holder_index=1, max_size=1000)
    def form_structure_product(self, particle_i: Particle, particle_j: Particle) -> np.ndarray:
        return self.form_profile(particle_i) * self.form_profile(particle_j) * self.structure_factor(particle_i, particle_j)

    def box_intensity(self, box: Box) -> np.ndarray:
        return (1e8 / box.volume) * array_list_sum(
            [array_list_sum(
                [self.form_structure_product(particle_i, particle_j) for particle_j in box.particles]
            ) for particle_i in box.particles]
        )
            

if __name__ == "__main__":
    pass

#%%
