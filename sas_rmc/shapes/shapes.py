from dataclasses import field, dataclass
from abc import ABC, abstractmethod
from __future__ import annotations

import numpy as np
from matplotlib import patches

from sas_rmc.vector import Vector, Interface





@dataclass
class Shape(ABC):
    """Abstract base class for a Shape type.

    """

    @abstractmethod
    def get_position(self) -> Vector:
        pass

    @abstractmethod
    def get_orientation(self) -> Vector:
        pass
    
    

    @abstractmethod
    def is_inside(self, position: Vector) -> bool:
        """Indicates whether a vector is inside the shape.

        This is an abstract method, so you will need to write your own implementation if you define a new Shape type. This is used for collision detection, but not for form factor calculations. Please use concrete class attributes for form factor calculations.

        Parameters
        ----------
        position : Vector
            The position vector to test.

        Returns
        -------
        bool
            Returns True if the position vector is inside the shape.
        """
        pass

    @property
    @abstractmethod
    def volume(self) -> float:
        """Calculates the volume of a shape.

        Returns
        -------
        float
            The volume of the shape.
        """
        pass

    @abstractmethod
    def closest_surface_position(self, position: Vector) -> Vector:
        """Returns the position on the surface of the shape which is closest to the input position Vector.

        Parameters
        ----------
        position : Vector
            The vector to test against.

        Returns
        -------
        Vector
            A position on the surface of the shape which is closest to the input position.
        """
        pass

    def collision_detected(self, shape: Shape) -> bool:
        """Tests whether another shape has collided with or impinged with this shape.

        Parameters
        ----------
        shape : Shape
            Another shape.

        Returns
        -------
        bool
            Returns True is a collision or impingement has been detected.
        """
        for shape_1, shape_2 in zip([self, shape], [shape, self]):
            closest_position_on_shape_2_surface = shape_2.closest_surface_position(shape_1.get_position())
            if shape_1.is_inside(closest_position_on_shape_2_surface):
                return True
        return False
        
    @abstractmethod
    def random_position_inside(self) -> Vector:
        """A random position which is inside the shape.

        This is an abstract method, so you will need to write your own implementation if you define a new Shape type.

        Returns
        -------
        Vector
            A randomly generated position vector which is inside the shape.
        """
        pass

    @abstractmethod
    def change_position(self, position: Vector) -> Shape:
        pass

    @abstractmethod
    def change_orientation(self, orientation: Vector) -> Shape:
        pass


@dataclass
class Interface:
    position_marker: Vector
    normal: Vector

    def is_inside(self, position: Vector) -> bool:
        return (position - self.position_marker) * self.normal < 0

    def on_surface(self, position: Vector) -> bool:
        return (position - self.position_marker) * self.normal == 0

    def project_onto_surface(self, position: Vector) -> bool:
        position_ref = position - self.position_marker
        return position_ref - (self.normal.unit_vector * position_ref) * self.normal.unit_vector + self.position_marker



    
@dataclass
class Cube(Shape):
    orientation: Vector = field(default_factory=lambda : Vector(0, 0, 1))
    dimension_0: float = 0
    dimension_1: float = 0
    dimension_2: float = 0

    @property
    def dimensions(self) -> tuple[float, float, float]:
        return self.dimension_0, self.dimension_1, self.dimension_2

    @property
    def volume(self) -> float:
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
        distances_to_surface = []
        positions_on_surface = []
        for interface in self.end_interfaces:
            position_on_surface = interface.project_onto_surface(position)
            distance_to_surface = (positions_on_surface - position).mag
            distances_to_surface.append(distance_to_surface)
            positions_on_surface.append(position_on_surface)
        return position_on_surface[np.argmin(distances_to_surface)]

    def random_position_inside(self) -> Vector:
        a, b, c = [np.random.uniform(low = -h/2, high = +h/2) for h in [self.dimension_0, self.dimension_1, self.dimension_2]]
        basis_c, basis_a, basis_b = self.orientation.rotated_basis()
        return self.central_position + a * basis_a + b * basis_b + c * basis_c

    def get_patches(self, **kwargs) -> patches.Rectangle:
        return patches.Rectangle(
            xy = (self.central_position.x - self.dimension_0 / 2, self.central_position.y - self.dimension_1 / 2),
            width = self.dimension_0,
            height = 2 * self.dimension_1,
            angle = np.arctan2(self.orientation.y, self.orientation.x),
            **kwargs
        )

    def change_position(self, position: Vector):
        return Cube(
            central_position=position,
            orientation=self.orientation,
            dimension_0=self.dimension_0,
            dimension_1=self.dimension_1,
            dimension_2=self.dimension_2
        )

    def change_orientation(self, orientation: Vector):
        return Cube(
            central_position=self.central_position,
            orientation=orientation,
            dimension_0=self.dimension_0,
            dimension_1=self.dimension_1,
            dimension_2=self.dimension_2
        )
    

def collision_detected(shapes_1: list[Shape], shape_2: list[Shape]) -> bool:
    """Detect if a collision has occured between two lists of shapes.

    I made this as a helper function for collision detection because it has less coupling than my first implementation.

    Parameters
    ----------
    shapes_1 : List[Shape]
        A list of shapes.
    shape_2 : List[Shape]
        Another list of shapes.

    Returns
    -------
    bool
        Returns True if any collision is detected between a shape in the first list and a shape in the second list
    """
    for shape in shapes_1:
        for other_shape in shape_2:
            if shape.collision_detected(other_shape):
                return True
    return False