from dataclasses import dataclass

import numpy as np
from typing_extensions import Self

from sas_rmc.shapes import Shape, Interface
from sas_rmc import Vector, constants

PI = constants.PI

def cylinder_volume(radius: float, height: float) -> float:
    return height * PI * (radius**2)


@dataclass
class Cylinder(Shape):
    radius: float
    height: float
    central_position: Vector
    orientation: Vector

    def get_position(self) -> Vector:
        return self.central_position
    
    def get_orientation(self) -> Vector:
        return self.orientation

    @property
    def end_interfaces(self) -> list[Interface]:
        orientation_unit_vector = self.orientation.unit_vector
        return [Interface(
            position_marker = self.central_position + f * (self.height / 2) * orientation_unit_vector, 
            normal=f * orientation_unit_vector
            ) for f in [-1, +1]]
        
        
    def _project_to_cylinder_axis(self, position: Vector) -> Vector: 
        interfaces = self.end_interfaces
        orientation_reffed = interfaces[1].position_marker - interfaces[0].position_marker
        position_reffed = position - interfaces[0].position_marker
        scalar_projection = position_reffed * (orientation_reffed.unit_vector)
        if scalar_projection < 0:
            return interfaces[0].position_marker
        if scalar_projection > orientation_reffed.mag:
            return interfaces[1].position_marker
        return scalar_projection * (orientation_reffed.unit_vector) + interfaces[0].position_marker

    def is_inside(self, position: Vector) -> bool:
        if all(interface.is_inside(position) for interface in self.end_interfaces):
            axis_position = self._project_to_cylinder_axis(position)
            return (position - axis_position).mag <= self.radius
        return False
    
    def get_volume(self) -> float:
        return cylinder_volume(self.radius, self.height)

    def closest_surface_position(self, position: Vector) -> Vector:
        position_copy = position.copy()
        for interface in self.end_interfaces:
            if not interface.is_inside(position_copy):
                position_copy = interface.project_onto_surface(position_copy)
        axis_projection = self._project_to_cylinder_axis(position_copy)
        pointing_vector = (position - axis_projection).unit_vector
        return self.radius * pointing_vector + axis_projection

    def random_position_inside(self) -> Vector:
        z = np.random.uniform(low = -self.height/2, high=+self.height/2)
        phi = np.random.uniform(low = 0, high = 2* PI )
        r = np.random.uniform(low = 0, high=self.radius)
        para_vec, radial_vec_1, radial_vec_2 = self.orientation.rotated_basis()
        return self.central_position + z * para_vec + r * (np.cos(phi) * radial_vec_1 + np.sin(phi) * radial_vec_2)

    def change_position(self, position: Vector) -> Self:
        return type(self)(
            central_position=position,
            orientation=self.orientation,
            radius=self.radius,
            height=self.height
        )

    def change_orientation(self, orientation: Vector) -> Self:
        return type(self)(
            central_position=self.central_position,
            orientation=orientation,
            radius=self.radius,
            height=self.height
        )

    def change_radius(self, radius: float) -> Self:
        return type(self)(
            central_position=self.central_position,
            orientation=self.orientation,
            radius = radius,
            height= self.height
        )
    
    def change_height(self, height: float) -> Self:
        return type(self)(
            central_position=self.central_position,
            orientation=self.orientation,
            radius=self.radius,
            height=height
        )
    

