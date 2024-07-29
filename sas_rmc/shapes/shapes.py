from dataclasses import dataclass
from abc import ABC, abstractmethod

from typing_extensions import Self

from sas_rmc.vector import Vector





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

    
    @abstractmethod
    def get_volume(self) -> float:
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

    def collision_detected(self, shape: Self) -> bool:
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
    def change_position(self, position: Vector) -> Self:
        pass

    @abstractmethod
    def change_orientation(self, orientation: Vector) -> Self:
        pass


@dataclass
class Interface:
    position_marker: Vector
    normal: Vector

    def is_inside(self, position: Vector) -> bool:
        return (position - self.position_marker) * self.normal < 0

    def on_surface(self, position: Vector) -> bool:
        return (position - self.position_marker) * self.normal == 0

    def project_onto_surface(self, position: Vector) -> Vector:
        position_ref = position - self.position_marker
        return position_ref - (self.normal.unit_vector * position_ref) * self.normal.unit_vector + self.position_marker



 
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

#%%
if __name__ == "__main__":
    interface = Interface(Vector(0, 0, 0), normal=Vector(0, 0, 1))

#%%