#%%
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from sas_rmc.particles import Particle, FormResult
from sas_rmc.particles.particle_spherical import SphericalParticle
from sas_rmc import Vector
from sas_rmc.shapes import Shape, Sphere, collision_detected

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
    
    @property
    def core_radius(self) -> float:
        return self.core_sphere.radius
    
    @property
    def thickness(self) -> float:
        return self.shell_sphere.radius - self.core_radius
    
    def validate_shape(self) -> None:
        if self.core_sphere.get_position() != self.shell_sphere.get_position():
            raise ValueError("Core shell particle position failure")
    
    def get_position(self) -> Vector:
        self.validate_shape()
        return self.core_sphere.get_position()
    
    def get_orientation(self) -> Vector:
        return self.core_sphere.get_orientation()

    def form_array(self, qx_array: np.ndarray, qy_array: np.ndarray) -> np.ndarray:
        core_particle = SphericalParticle(
            core_sphere=self.core_sphere,
            sphere_sld=self.core_sld - self.shell_sld,
            solvent_sld=self.solvent_sld,
            magnetization=Vector.null_vector()
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
    
    def get_shapes(self) -> list[Shape]:
        return [self.core_sphere, self.shell_sphere]
    
    def is_inside(self, position: Vector) -> bool:
        self.validate_shape()
        return self.shell_sphere.is_inside(position)
    
    def collision_detected(self, other_particle: Particle) -> bool:
        self.validate_shape()
        return collision_detected([self.shell_sphere], other_particle.get_shapes)
    
    def get_scattering_length(self) -> float:
        return (self.core_sld - self.shell_sld) * self.core_sphere.get_volume() + self.shell_sld * self.shell_sphere.get_volume()

    def change_position(self, position: Vector) -> CoreShellParticle:
        return CoreShellParticle.gen_from_parameters(
            position=position,
            magnetization=self.magnetization,
            core_radius=self.core_radius,
            thickness=self.thickness,
            core_sld=self.core_sld,
            shell_sld=self.shell_sld,
            solvent_sld=self.solvent_sld
        )
    
    def change_magnetization(self, magnetization: Vector) -> CoreShellParticle:
        return CoreShellParticle.gen_from_parameters(
            position=self.get_position(),
            magnetization=magnetization,
            core_radius=self.core_radius,
            shell_sld=self.shell_sld,
            solvent_sld=self.solvent_sld
        )


if __name__ == "__main__":
    test = CoreShellParticle.gen_from_parameters(position= Vector(0,0,1))

#%%
