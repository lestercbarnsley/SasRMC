#%%
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .particle import Particle, modulus_array, form_array_sphere, magnetic_sld_in_angstrom_minus_2
from ..vector import Vector
from ..shapes.shapes import Shape, Sphere, collision_detected


@dataclass
class CoreShellParticle(Particle):
    _shapes: List[Sphere] = field(
        default_factory=lambda : [Sphere(), Sphere()]
    )
    core_sld: float = 0
    shell_sld: float = 0

    @classmethod
    def gen_from_parameters(cls, position: Vector, magnetization: Vector = None, core_radius: float = 0, thickness: float = 0, core_sld: float = 0, shell_sld: float = 0, solvent_sld: float = 0):
        sphere_inner = Sphere(central_position=position, radius=core_radius)
        sphere_outer = Sphere(central_position=position, radius= core_radius + thickness)
        if magnetization is None:
            magnetization = Vector.null_vector()
        return cls(
            _magnetization=magnetization,
            _shapes = [sphere_inner, sphere_outer],
            core_sld=core_sld,
            shell_sld=shell_sld,
            solvent_sld=solvent_sld
            )

    @property
    def shapes(self) -> List[Sphere]:
        return super().shapes

    def _change_shapes(self, shape_list: List[Shape]):
        return CoreShellParticle(
            _magnetization = self._magnetization,
            _shapes = shape_list,
            solvent_sld=self.solvent_sld,
            core_sld=self.core_sld,
            shell_sld=self.shell_sld
        )
    
    def is_spherical(self) -> bool:
        return True

    @property
    def core_sphere(self) -> Sphere:
        return self.shapes[0]
    
    @property
    def shell_sphere(self) -> Sphere:
        return self.shapes[1]

    @property
    def shell_thickness(self) -> float:
        core_radius = self.core_sphere.radius
        overall_radius = self.shell_sphere.radius
        return overall_radius - core_radius

    def is_inside(self, position: Vector) -> bool:
        return self.shell_sphere.is_inside(position)

    @property
    def volume(self) -> float:
        return self.shell_sphere.volume

    @property
    def scattering_length(self):
        core_sphere = self.core_sphere
        shell_sphere = self.shell_sphere
        return self.delta_sld(self.core_sld - self.shell_sld) * core_sphere.volume + self.delta_sld(self.shell_sld) * shell_sphere.volume
 
    @property
    def position(self) -> Vector:
        return super().position

    def set_position(self, position: Vector) -> Particle:
        spheres = [sphere.change_position(position) for sphere in self.shapes]
        return self._change_shapes(spheres)

    def set_magnetization(self, magnetization: Vector) -> Particle:
        return CoreShellParticle.gen_from_parameters(
            position=self.position,
            magnetization=magnetization,
            core_radius=self.core_sphere.radius,
            thickness=self.shell_thickness,
            core_sld=self.core_sld,
            shell_sld=self.shell_sld,
            solvent_sld=self.solvent_sld
        )

    def form_array(self, qx_array: np.ndarray, qy_array: np.ndarray, orientation: Vector) -> np.ndarray:
        q = modulus_array(qx_array, qy_array)
        core_form = form_array_sphere(
            radius = self.core_sphere.radius, 
            sld = self.delta_sld(self.core_sld) - self.delta_sld(self.shell_sld), 
            q_array=q)
        shell_form = form_array_sphere(
            radius = self.shell_sphere.radius, 
            sld = self.delta_sld(self.shell_sld), 
            q_array = q)
        return core_form  + shell_form

    def magnetic_form_array(self, qx_array: np.ndarray, qy_array: np.ndarray, orientation: Vector, magnetization: Vector) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = modulus_array(qx_array, qy_array)
        if not self.is_magnetic():
            return [np.zeros(q.shape) for _ in range(3)]
        return [form_array_sphere(self.core_sphere.radius, magnetic_sld, q) for magnetic_sld in magnetic_sld_in_angstrom_minus_2(magnetization)]

    def get_sld(self, relative_position: Vector) -> float:
        position = relative_position + self.position
        if self.core_sphere.is_inside(position):
            return self.core_sld
        if self.is_inside(position):
            return self.shell_sld
        return self.solvent_sld

    def get_magnetization(self, relative_position: Vector) -> Vector:
        position = relative_position + self.position
        return self.magnetization if self.core_sphere.is_inside(position) else Vector.null_vector()

    def collision_detected(self, other_particle: Particle) -> bool:
        biggest_shape = self.shell_sphere
        return collision_detected([biggest_shape], other_particle.shapes)

    def closest_surface_position(self, position: Vector) -> Vector:
        return self.shell_sphere.closest_surface_position(position)

    def get_average_sld(self, radius: float) -> float:
        if radius < self.core_sphere.radius:
            return self.core_sld
        if radius < self.shell_sphere.radius:
            return self.shell_sld
        return self.solvent_sld

    def get_loggable_data(self) -> dict:
        data = {
            'Particle type': "",
            'Core radius': self.core_sphere.radius,
            'Shell thickness': self.shell_thickness,
            'Core SLD': self.core_sld,
            'Shell SLD' : self.shell_sld,
            'Solvent SLD': self.solvent_sld,
        }
        data.update(super().get_loggable_data())
        return data

    


if __name__ == "__main__":
    pass

#%%
