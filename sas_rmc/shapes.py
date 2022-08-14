from typing import List, Tuple
from dataclasses import field, dataclass
from abc import ABC, abstractmethod

import numpy as np
#from matplotlib import pyplot as plt
from matplotlib import patches

from .vector import Vector, Interface

PI = np.pi

sphere_volume = lambda radius: (4 *PI / 3) * (radius**3)


@dataclass
class Shape(ABC):
    """Abstract base class for a Shape type.

    Attributes
    ----------
    central_position : Vector, optional
        A vector to represent the position of this shape (usually the central position).
    orientation : Vector, optional
        A vector describing the orientation of the particle. This matters for anisotropic (non-spherical) shapes.

    """
    central_position: Vector = field(default_factory=Vector.null_vector)
    orientation: Vector = field(default_factory=lambda : Vector(0,1,0))

    @property
    @abstractmethod
    def dimensions(self) -> Tuple[float, float, float]:
        """Get a tuple of float values indicating the relative dimensions of the shape.
        
        This is used to give guidance to other methods about physical boundaries required to encapsulate the shape, i.e., constructing a VectorSpace that entire encloses the shape 

        Returns
        -------
        Tuple[float, float, float]
            A tuple of three floats with the total extent of the shape in each dimension.
        """

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

    def collision_detected(self, shape) -> bool:
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
        return self.is_inside(shape.closest_surface_position(self.central_position)) or shape.is_inside(self.closest_surface_position(shape.central_position))

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
    def get_patches(self, **kwargs) -> patches.Patch:
        pass

    @abstractmethod
    def change_position(self, position: Vector):
        pass

    @abstractmethod
    def change_orientation(self, orientation: Vector):
        pass


@dataclass
class Sphere(Shape):
    radius: float = 0

    @property
    def dimensions(self) -> Tuple[float, float, float]:
        return 2 * self.radius, 2 * self.radius, 2 * self.radius

    def is_inside(self, position: Vector) -> bool:
        return (position - self.central_position).mag <= self.radius

    @property
    def volume(self) -> float:
        return sphere_volume(self.radius)

    def closest_surface_position(self, position: Vector) -> Vector:
        pointing_vector = (position - self.central_position).unit_vector
        return (self.radius * pointing_vector) + self.central_position

    def random_position_inside(self) -> Vector:
        return Vector.random_normal_vector(np.random.uniform(low=0, high=self.radius))

    def get_patches(self, **kwargs) -> patches.Circle:
        return patches.Circle(
            xy = (self.central_position.x, self.central_position.y),
            radius = self.radius,
            **kwargs
        )

    def change_position(self, position: Vector):
        return Sphere(central_position=position,radius = self.radius)

    def change_orientation(self, orientation: Vector):
        return self

@dataclass
class Cylinder(Shape):
    radius: float = 0
    height: float = 0

    @property
    def dimensions(self) -> Tuple[float, float, float]:
        return 2 * self.radius, 2 * self.radius, self.height

    @property
    def end_interfaces(self) -> List[Interface]:
        def define_interface(plus_or_minus: bool) -> Interface:
            factor = +1 if plus_or_minus else -1
            position_marker = self.central_position + factor * (self.height / 2) * (self.orientation.unit_vector)
            return Interface(position_marker, normal = factor * self.orientation.unit_vector)
        return [define_interface(plus_or_minus) for plus_or_minus in [True, False]]
        
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

    @property
    def volume(self) -> float:
        return self.height * PI * (self.radius**2)

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

    def get_patches(self, **kwargs) -> patches.Rectangle:
        return patches.Rectangle(
            xy = (self.central_position.x - self.radius, self.central_position.y - self.height / 2),
            width = self.height,
            height = 2 * self.radius,
            angle = np.arctan2(self.orientation.y, self.orientation.x),
            **kwargs
        )

    def change_position(self, position: Vector):
        return type(self)(
            central_position=position,
            orientation=self.orientation,
            radius=self.radius,
            height=self.height
        )

    def change_orientation(self, orientation: Vector):
        return type(self)(
            central_position=self.central_position,
            orientation=orientation,
            radius=self.radius,
            height=self.height
        )
    
@dataclass
class Cube(Shape):
    orientation: Vector = field(default_factory=lambda : Vector(0, 0, 1))
    dimension_0: float = 0
    dimension_1: float = 0
    dimension_2: float = 0

    @property
    def dimensions(self) -> Tuple[float, float, float]:
        return self.dimension_0, self.dimension_1, self.dimension_2

    @property
    def volume(self) -> float:
        return self.dimension_0 * self.dimension_1 * self.dimension_2

    @property
    def end_interfaces(self) -> List[Interface]:
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
    

def collision_detected(shapes_1: List[Shape], shape_2: List[Shape]) -> bool:
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