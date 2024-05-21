from dataclasses import dataclass
from __future__ import annotations

import numpy as np

from sas_rmc.shapes import Shape
from sas_rmc import Vector, constants

PI = constants.PI

def sphere_volume(radius: float) -> float:
    return (4 *PI / 3) * (radius**3)


@dataclass
class Sphere(Shape):
    radius: float
    central_position: Vector

    def get_position(self) -> Vector:
        return self.central_position
    
    def get_orientation(self) -> Vector:
        return Vector(0, 1, 0)
    
    def is_inside(self, position: Vector) -> bool:
        return (position - self.central_position).mag <= self.radius

    @property
    def volume(self) -> float:
        return sphere_volume(self.radius)

    def closest_surface_position(self, position: Vector) -> Vector:
        pointing_vector = (position - self.central_position).unit_vector
        return (self.radius * pointing_vector) + self.central_position

    def random_position_inside(self) -> Vector:
        return Vector.random_normal_vector(np.random.uniform(low=0, high=self.radius)) + self.central_position

    def change_position(self, position: Vector) -> Sphere:
        return Sphere(
            radius=self.radius,
            central_position=position
        )

    def change_orientation(self, orientation: Vector) -> Sphere:
        return self

    def change_radius(self, radius: float) -> Sphere:
        return Sphere(
            radius=radius,
            central_position=self.central_position
        )