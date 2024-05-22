#%%
from dataclasses import dataclass

import numpy as np

from sas_rmc.particles import Particle, FormResult
from sas_rmc.particles.particle_spherical import SphericalParticle, form_array_sphere
from sas_rmc import Vector
from sas_rmc.shapes import Sphere

@dataclass
class CoreShellParticle(Particle):
    core_sphere: Sphere
    shell_sphere: Sphere
    core_sld: float
    shell_sld: float
    solvent_sld: float
    magnetization: Vector

    @classmethod
    def gen_from_parameters(cls, position: Vector, magnetization: Vector | None = None, core_radius: float = 0, thickness: float = 0, core_sld: float = 0, shell_sld: float = 0, solvent_sld: float = 0):
        return CoreShellParticle(
            core_sphere=Sphere(radius=core_radius, central_position=position),
            shell_sphere=Sphere(radius=core_radius + thickness, central_position=position),
            core_sld=core_sld,
            shell_sld=shell_sld,
            solvent_sld=solvent_sld,
            magnetization=magnetization
        )
    
    def get_position(self) -> Vector:
        if self.core_sphere.get_position() != self.shell_sphere.get_position():
            raise ValueError("Core shell particle position failure")
        return self.core_sphere.get_position()
    
    def get_orientation(self) -> Vector:
        return self.core_sphere.get_orientation()
    
    def form_array(self, qx_array: np.ndarray, qy_array: np.ndarray) -> np.ndarray:
        core_particle = SphericalParticle(
            core_sphere=self.core_sphere,
            sphere_sld=self.core_sld - self.shell_sld,
            solvent_sld=self.solvent_sld,
            magnetization=self.magnetization
        )
        shell_particle = SphericalParticle(
            core_sphere=self.shell_sphere,
            sphere_sld=self.shell_sld,
            solvent_sld=self.solvent_sld,
            magnetization=Vector.null_vector()
        )
        return core_particle.form_array(qx_array, qy_array) + shell_particle.form_array(qx_array, qy_array)
    
    def magnetic_form_array(self, qx_array: np.ndarray, qy_array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        spherical_magnetic_particle = SphericalParticle(
            core_sphere=self.core_sphere,
            sphere_sld=self.core_sld,
            solvent_sld=self.solvent_sld,
            magnetization=self.magnetization
        )
        return spherical_magnetic_particle.magnetic_form_array(qx_array, qy_array)
    
    def form_result(self, qx_array: np.ndarray, qy_array: np.ndarray) -> FormResult:
        form_nuclear = self.form_array(qx_array, qy_array)
        form_mag_x, form_mag_y, form_mag_z = self.magnetic_form_array(qx_array, qy_array)
        return FormResult(
            form_nuclear=form_nuclear,
            form_magnetic_x=form_mag_x,
            form_magnetic_y=form_mag_y,
            form_magnetic_z=form_mag_z
        )


if __name__ == "__main__":
    pass

#%%
