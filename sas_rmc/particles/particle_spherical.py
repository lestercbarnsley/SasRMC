#%%
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .particle import Particle, modulus_array, form_array_sphere, magnetic_sld_in_angstrom_minus_2
from ..vector import Vector
from ..shapes.shapes import Shape, Sphere


@dataclass
class SphericalParticle(Particle):
    _shapes: List[Sphere] = field(default_factory=lambda : [Sphere()])
    sphere_sld: float = 0

    @classmethod
    def gen_from_parameters(cls, position: Vector, magnetization: Vector = None, sphere_radius: float = 0, sphere_sld: float = 0, solvent_sld: float = 0):
        return cls(
            _magnetization = magnetization,
            _shapes = [Sphere(
                central_position=position,
                radius=sphere_radius
                )],
            solvent_sld=solvent_sld,
            sphere_sld=sphere_sld
        )

    @property
    def shapes(self) -> List[Sphere]:
        return super().shapes

    def _change_shapes(self, shape_list: List[Shape]):
        return type(self)(
            _magnetization = self._magnetization,
            _shapes = shape_list,
            solvent_sld=self.solvent_sld,
            sphere_sld=self.sphere_sld
        )

    @property
    def volume(self) -> float:
        return self.shapes[0].volume

    def is_spherical(self) -> bool:
        return True

    @property
    def scattering_length(self) -> float:
        return self.delta_sld(self.sphere_sld) * self.volume

    def set_position(self, position: Vector) -> Particle:
        sphere_list = [sphere.change_position(position) for sphere in self.shapes]
        return self._change_shapes(sphere_list)

    def set_magnetization(self, magnetization: Vector) -> Particle:
        return type(self).gen_from_parameters(
            position=self.position,
            magnetization=magnetization,
            sphere_radius=self.shapes[0].radius,
            sphere_sld=self.sphere_sld,
            solvent_sld=self.solvent_sld
        ) 

    def form_array(self, qx_array: np.ndarray, qy_array: np.ndarray, orientation: Vector) -> np.ndarray:
        q = modulus_array(qx_array, qy_array)
        return form_array_sphere(
            radius = self.shapes[0].radius, 
            sld = self.delta_sld(self.sphere_sld), 
            q_array=q)

    def magnetic_form_array(self, qx_array: np.ndarray, qy_array: np.ndarray, orientation: Vector, magnetization: Vector) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = modulus_array(qx_array, qy_array)
        if not self.is_magnetic():
            return [np.zeros(q.shape) for _ in range(3)]
        sphere = self.shapes[0]
        return [form_array_sphere(sphere.radius, magnetic_sld, q) for magnetic_sld in magnetic_sld_in_angstrom_minus_2(magnetization)]

    def get_sld(self, relative_position: Vector) -> float:
        position = relative_position + self.position
        return self.sphere_sld if self.is_inside(position) else self.solvent_sld

    def get_magnetization(self, relative_position: Vector) -> Vector:
        position = relative_position + self.position
        return self.magnetization if self.is_inside(position) else self.solvent_sld

    def closest_surface_position(self, position: Vector) -> Vector:
        return self.shapes[0].closest_surface_position(position)

    def get_average_sld(self, radius: float) -> float:
        if radius < self.shapes[0].radius:
            return self.sphere_sld
        return self.solvent_sld


if __name__ == "__main__":
    pass

#%%
