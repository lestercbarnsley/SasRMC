#%%
from dataclasses import dataclass

import numpy as np
from typing_extensions import Self

from sas_rmc import Vector
from sas_rmc.array_cache import array_cache
from sas_rmc.particles.particle import FormResult, Particle, magnetic_sld_in_angstrom_minus_2
from sas_rmc.particles.particle_form import ParticleArray
from sas_rmc.particles.particle_profile import ParticleProfile
from sas_rmc.shapes.sphere import sphere_volume, Sphere


def theta_fn(qR: np.ndarray) -> np.ndarray:
    return np.where(qR == 0, 1, 3 * (np.sin(qR) - qR* np.cos(qR)) / (qR**3))

@array_cache(max_size = 2_000)
def form_array_sphere_normalized(radius: float, q_array: np.ndarray) -> np.ndarray:
    volume = sphere_volume(radius)
    theta_arr = theta_fn(q_array * radius)
    return volume * theta_arr

@array_cache(max_size = 2_000)
def form_array_sphere(radius: float, sld: float, q_array: np.ndarray) -> np.ndarray:
    volume_theta_product = form_array_sphere_normalized(radius, q_array)
    return sld * volume_theta_product

@array_cache(max_size=50)
def array_magnitude(*q_component_arrays: np.ndarray) -> np.ndarray: # I'm considering moving this to vector module
    return np.sqrt((np.array(q_component_arrays)**2).sum(axis = 0)) 
    # I'm fighting numpy's type hinting limitations here, but it's a short function with properly types inputs and outputs


@dataclass
class SphericalParticle(Particle):
    core_sphere: Sphere
    sphere_sld: float
    solvent_sld: float
    magnetization: Vector

    @classmethod
    def gen_from_parameters(cls, position: Vector, magnetization: Vector | None = None, sphere_radius: float = 0, sphere_sld: float = 0, solvent_sld: float = 0):
        return SphericalParticle(
            core_sphere=Sphere(radius=sphere_radius, central_position=position),
            sphere_sld=sphere_sld,
            solvent_sld=solvent_sld,
            magnetization=magnetization if magnetization is not None else Vector.null_vector()
        )

    def get_delta_sld(self) -> float:
        return (self.sphere_sld - self.solvent_sld) * 1e-6
    
    def form_array(self, qx_array: np.ndarray, qy_array: np.ndarray) -> np.ndarray:
        q_array = array_magnitude(qx_array, qy_array)
        delta_sld = self.get_delta_sld()
        return form_array_sphere(self.core_sphere.radius, delta_sld, q_array)
    
    def get_position(self) -> Vector:
        return self.core_sphere.get_position()
    
    def get_orientation(self) -> Vector:
        return self.core_sphere.get_orientation()
    
    def get_magnetization(self) -> Vector:
        return self.magnetization
    
    def get_volume(self) -> float:
        return self.core_sphere.get_volume()
    
    def magnetic_form_array(self, qx_array: np.ndarray, qy_array: np.ndarray) -> list[np.ndarray]:
        q = array_magnitude(qx_array, qy_array)
        if not self.is_magnetic():
            return [np.zeros(q.shape) for _ in range(3)]
        sphere = self.core_sphere
        return [form_array_sphere(sphere.radius, magnetic_sld, q) for magnetic_sld in magnetic_sld_in_angstrom_minus_2(self.magnetization)]

    def form_result(self, qx_array: np.ndarray, qy_array: np.ndarray) -> FormResult:
        form_nuclear = self.form_array(qx_array, qy_array)
        form_mag_x, form_mag_y, form_mag_z = self.magnetic_form_array(qx_array, qy_array)
        return FormResult(
            form_nuclear=form_nuclear,
            form_magnetic_x=form_mag_x,
            form_magnetic_y=form_mag_y,
            form_magnetic_z=form_mag_z
        )

    def get_shapes(self) -> list[Sphere]:
        return [self.core_sphere]
    
    def get_scattering_length(self) -> float:
        return self.get_volume() * self.get_delta_sld()
    
    def change_position(self, position: Vector) -> Self:
        return type(self)(
            core_sphere=self.core_sphere.change_position(position),
            sphere_sld=self.sphere_sld,
            solvent_sld=self.solvent_sld,
            magnetization=self.magnetization
        )
    
    def change_orientation(self, orientation: Vector) -> Self:
        return self
    
    def change_magnetization(self, magnetization: Vector) -> Self:
        return type(self)(
            core_sphere=self.core_sphere,
            sphere_sld=self.sphere_sld,
            solvent_sld=self.solvent_sld,
            magnetization=magnetization
        )
    
    def get_loggable_data(self) -> dict:
        loggable_data = super().get_loggable_data()
        return loggable_data | {
            "Core radius" : self.core_sphere.radius,
            "Sphere SLD" : self.sphere_sld,
            "Solvent SLD" : self.solvent_sld,
        }
    

@dataclass
class SphericalParticleForm(ParticleArray):
    spherical_particle: SphericalParticle

    def get_bound_particle(self) -> SphericalParticle:
        return self.spherical_particle
    
    def form_result(self, qx_array: np.ndarray, qy_array: np.ndarray) -> FormResult:
        return self.get_bound_particle().form_result(qx_array, qy_array)
    
    def get_loggable_data(self) -> dict:
        return self.get_bound_particle().get_loggable_data()
    

@dataclass
class SphericalParticleProfile(ParticleProfile):
    spherical_particle: SphericalParticle

    def get_bound_particle(self) -> SphericalParticle:
        return self.spherical_particle
    
    def form_profile(self, q_profile: np.ndarray) -> np.ndarray:
        return self.get_bound_particle().form_array(q_profile, q_profile * 0)
    
    def get_loggable_data(self) -> dict:
        return self.get_bound_particle().get_loggable_data()
    


if __name__ == "__main__":
    arrs = [np.random.rand(100, 200) for i in range(3)]
    q_arr = array_magnitude(arrs[0], arrs[1], arrs[2])
    print(q_arr.shape)

    assert np.all(array_magnitude(arrs[0], arrs[1]) - np.sqrt(arrs[0]**2 + arrs[1]**2) == 0) #I'll use this for  a unit test later
    
  #%%
