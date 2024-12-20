#%%

from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from typing_extensions import Self

from sas_rmc.particles import FormResult, Particle
from sas_rmc.particles.particle_form import ParticleArray
from sas_rmc.particles.particle_spherical import SphericalParticle
from sas_rmc import Vector
from sas_rmc.shapes import Shape, Sphere, collision_detected


S = TypeVar("S", bound=Shape)

def change_shape_positions(shape_list: list[S], position: Vector) -> list[S]:
    return [shape.change_position(position) for shape in shape_list]

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
        return cls(
            core_sphere=Sphere(radius=core_radius, central_position=position),
            shell_sphere=Sphere(radius=core_radius + thickness, central_position=position),
            core_sld=core_sld,
            shell_sld=shell_sld,
            solvent_sld=solvent_sld,
            magnetization=magnetization if magnetization is not None else Vector.null_vector()
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
    
    def get_magnetization(self) -> Vector:
        return self.magnetization
    
    def get_volume(self) -> float:
        return self.shell_sphere.get_volume()

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
    
    def magnetic_form_array(self, qx_array: np.ndarray, qy_array: np.ndarray) -> list[np.ndarray]:
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
        return collision_detected([self.shell_sphere], other_particle.get_shapes())
    
    def get_scattering_length(self) -> float:
        return (self.core_sld - self.shell_sld) * self.core_sphere.get_volume() + self.shell_sld * self.shell_sphere.get_volume()

    def change_position(self, position: Vector) -> Self:
        core_sphere, shell_sphere = change_shape_positions([self.core_sphere, self.shell_sphere], position)
        return type(self)(
            core_sphere=core_sphere,
            shell_sphere=shell_sphere,
            core_sld=self.core_sld,
            shell_sld=self.shell_sld,
            solvent_sld=self.solvent_sld,
            magnetization=self.magnetization
        )
    
    def change_orientation(self, orientation: Vector) -> Self:
        return self

    def change_magnetization(self, magnetization: Vector) -> Self:
        return type(self)(
            core_sphere=self.core_sphere,
            shell_sphere=self.shell_sphere,
            core_sld=self.core_sld,
            shell_sld=self.shell_sld,
            solvent_sld=self.solvent_sld,
            magnetization=magnetization
        )
    
    def get_loggable_data(self) -> dict:
        loggable_data = super().get_loggable_data()
        return loggable_data | {
            "Core radius" : self.core_radius,
            "Thickness" : self.thickness,
            "Core SLD" : self.core_sld,
            "Shell SLD" : self.shell_sld,
            "Solvent SLD" : self.solvent_sld,
        }
    

@dataclass
class CoreShellParticleForm(ParticleArray):
    core_shell_particle: CoreShellParticle

    @classmethod
    def gen_from_parameters(cls, position: Vector, magnetization: Vector | None = None, core_radius: float = 0, thickness: float = 0, core_sld: float = 0, shell_sld: float = 0, solvent_sld: float = 0):
        core_shell_particle = CoreShellParticle.gen_from_parameters(
            position=position,
            magnetization=magnetization,
            core_radius=core_radius,
            thickness=thickness,
            core_sld=core_sld,
            shell_sld=shell_sld,
            solvent_sld=solvent_sld
        )
        return cls(core_shell_particle=core_shell_particle)

    def get_bound_particle(self) -> Particle:
        return self.core_shell_particle
    
    def change_bound_particle(self, bound_particle: Particle) -> Self:
        if not isinstance(bound_particle, CoreShellParticle):
            raise TypeError("Only bind core shell particle to this form calculator")
        return type(self)(core_shell_particle=bound_particle)
    
    def form_result(self, qx_array: np.ndarray, qy_array: np.ndarray) -> FormResult:
        return self.core_shell_particle.form_result(qx_array, qy_array)
    
    def get_loggable_data(self) -> dict:
        return self.get_bound_particle().get_loggable_data()
    

if __name__ == "__main__":
    t = CoreShellParticle.gen_from_parameters(
        position=Vector(0,0,0),
        core_radius=50,
        thickness=10,
        core_sld=6,
        shell_sld=1
    )
    p_1 = CoreShellParticle.gen_from_parameters(
        position=Vector(-1104.023002,4529.279886),
        core_radius = 132.071905,
        thickness=13.4,
        core_sld=6,
        shell_sld=1
    )
    p_2 = CoreShellParticle.gen_from_parameters(
        position=Vector(-1062.928088, 4823.94499),
        core_radius = 138.6449959,
        thickness=13.4,
        core_sld=6,
        shell_sld=1
    )
    print(p_1.collision_detected(p_2))

    core_shell_form = CoreShellParticleForm(core_shell_particle=p_2)

    print(core_shell_form.get_loggable_data())

#%%
