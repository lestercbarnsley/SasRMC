from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from sas_rmc.shapes import Shape, Interface
from sas_rmc import Vector


@dataclass
class Cube(Shape):
    central_position: Vector
    orientation: Vector
    dimension_0: float
    dimension_1: float
    dimension_2: float

    def get_position(self) -> Vector:
        return self.central_position
    
    def get_orientation(self) -> Vector:
        return self.orientation
    
    def get_volume(self) -> float:
        return self.dimension_0 * self.dimension_1 * self.dimension_2

    @property
    def end_interfaces(self) -> list[Interface]:
        central_position = self.central_position
        basis_c, basis_a, basis_b = self.orientation.rotated_basis()
        interfaces = []
        for h, basis in zip([self.dimension_0, self.dimension_1, self.dimension_2], [basis_a, basis_b, basis_c]):
            for m in [-1, +1]:
                position_marker = central_position + m * (h / 2) * basis
                normal = m * basis
                interfaces.append(Interface(position_marker, normal))
        return interfaces

    def is_inside(self, position: Vector) -> bool:
        return all(interface.is_inside(position) for interface in self.end_interfaces)

    def closest_surface_position(self, position: Vector) -> Vector:
        positions_on_surface = [interface.project_onto_surface(position) for interface in self.end_interfaces]
        return min(positions_on_surface, key=lambda vec : vec.distance_from_vector(position))

    def random_position_inside(self) -> Vector:
        a, b, c = [np.random.uniform(low = -h/2, high = +h/2) for h in [self.dimension_0, self.dimension_1, self.dimension_2]]
        basis_c, basis_a, basis_b = self.orientation.rotated_basis()
        return self.central_position + a * basis_a + b * basis_b + c * basis_c
    
    def change_position(self, position: Vector) -> Cube:
        return Cube(
            central_position=position,
            orientation=self.orientation,
            dimension_0=self.dimension_0,
            dimension_1=self.dimension_1,
            dimension_2=self.dimension_2
        )

    def change_orientation(self, orientation: Vector) -> Cube:
        return Cube(
            central_position=self.central_position,
            orientation=orientation,
            dimension_0=self.dimension_0,
            dimension_1=self.dimension_1,
            dimension_2=self.dimension_2
        )
    
    def change_dimensions(self, dimension_0: float, dimension_1: float, dimension_2: float) -> Cube:
        return Cube(
            central_position=self.central_position,
            orientation=self.orientation,
            dimension_0=dimension_0,
            dimension_1=dimension_1,
            dimension_2=dimension_2
        )
    
