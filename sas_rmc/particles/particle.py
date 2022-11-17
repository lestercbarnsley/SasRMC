#%%
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from scipy import constants

from ..array_cache import round_vector, array_cache
from ..vector import Vector
from ..shapes.shapes import Shape, Sphere, collision_detected
from .. import constants


PI = constants.PI
GAMMA_N = constants.GAMMA_N#np.abs(get_physical_constant('neutron mag. mom. to nuclear magneton ratio')) # This value is unitless
R_0 = constants.R_0# get_physical_constant('classical electron radius')
BOHR_MAG = constants.BOHR_MAG #get_physical_constant('Bohr magneton')
B_H_IN_INVERSE_AMP_METRES = constants.B_H_IN_INVERSE_AMP_METRES# (GAMMA_N * R_0 / 2) / BOHR_MAG


def magnetic_sld_in_angstrom_minus_2(magnetization_vector_in_amp_per_metre: Vector) -> Tuple[float, float, float]:
    # Let us do all calculations in metres, then convert to Angstrom^-2 as the last step
    magnetization = magnetization_vector_in_amp_per_metre
    sld_vector = B_H_IN_INVERSE_AMP_METRES * magnetization / (1e10**2)
    return sld_vector.x, sld_vector.y, sld_vector.z


sphere_volume = lambda radius: (4 * PI / 3) * radius**3
theta = lambda qR: np.where(qR == 0, 1, 3 * (np.sin(qR) - qR* np.cos(qR)) / (qR**3))
#modulus_array = lambda x_arr, y_arr: np.sqrt(x_arr**2 + y_arr**2)

@array_cache(max_size=5_000)
def modulus_array(x_arr: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
    return np.sqrt(x_arr**2 + y_arr**2)


@dataclass
class Particle(ABC):
    """Abstract base class for Particle type.

    When you make your own particle type, you should inherit from this class. This is an abstract class, so it cannot be instantiated itself, but it shows a template of how a Particle class should be written, and provides abstract methods that should have implementations written by the user.

    Attributes
    ----------
    shapes : List[Shape]
        List of shapes used to calculate collision detection.
    solvent_sld : float, optional
        The SLD of the solvent phase in 1E-6 * Angstrom^-2.
    _magnetization : Vector, optional
        A vector describing the magnetization of the magnetic phase. Units are Amps/metre.

    """
    _magnetization: Vector = field(default_factory = Vector.null_vector)
    _shapes: List[Shape] = field(default_factory = list)
    solvent_sld: float = 0

    @property
    def shapes(self) -> List[Shape]:
        return self._shapes

    @abstractmethod
    def _change_shapes(self, shape_list: List[Shape]):
        pass

    @property
    @abstractmethod
    def volume(self) -> float:
        """The volume of the particle.

        Returns
        -------
        float
            The volume of the particle in units Angstrom^3
        """
        return np.sum([shape.volume for shape in self.shapes])

    @property
    @abstractmethod
    def scattering_length(self) -> float:
        pass

    def is_magnetic(self) -> bool:
        """Is the particle magnetic?

        Returns
        -------
        bool
            Returns True if the particle is magnetic.
        """
        return self._magnetization.mag != 0
    
    @property
    def position(self) -> Vector:
        """A vector to represent the position of the particle.

        Usually this is the centre of the particle. This may also be the centre of one of the Shape objects in self.shapes.

        Returns
        -------
        Vector
            The position of the particle in Angstroms.
        """
        return self.shapes[0].central_position

    def set_position(self, position: Vector):
        """Set the position vector of a Particle object.

        Parameters
        ----------
        position : Vector
            The new position vector of the particle, in Angstrom.
        """
        position_change = position - self.position
        new_shapes = [shape.change_position(shape.central_position + position_change) for shape in self.shapes]
        return self._change_shapes(new_shapes)

    def is_spherical(self) -> bool: # Don't allow orientation changes to a spherical particle
        """Check if the particle is spherical.

        In this context, a particle is spherical if an orientation change has no effect on the particle. Therefore, a particle that consists of a group of spheres may not be spherical itself, but it would be if all these particles share the same centre position. This is essentially how the default implementation works, but you are STRONGLY encouraged to write your own implementation when you make a new Particle type (even though this isn't an abstract method). After all, you yourself know if your particle is spherical or not. You do know if your particle is spherical... right?

        Returns
        -------
        bool
            Returns True if the particle is spherical
        """
        same_location = lambda : all(round_vector(shape.central_position) == round_vector(self.shapes[0].central_position) for shape in self.shapes)
        return all(isinstance(shape, Sphere) for shape in self.shapes) and same_location()

    @property
    def orientation(self) -> Vector:
        return self.shapes[0].orientation

    def set_orientation(self, orientation: Vector):
        if self.is_spherical():
            return self
        new_shapes = [shape.change_orientation(orientation) for shape in self.shapes]
        return self._change_shapes(new_shapes)

    @property
    def magnetization(self) -> Vector:
        return self._magnetization

    @abstractmethod
    def set_magnetization(self, magnetization: Vector):
        pass

    def is_inside(self, position: Vector) -> bool:
        """Method for determining if a position in space is inside a particle.

        Parameters
        ----------
        position : Vector
            Position vector to test.

        Returns
        -------
        bool
            Returns True if the position is inside the particle.
        """
        return any(shape.is_inside(position) for shape in self.shapes) # Use a generator here to take advantage of lazy iteration
        
        
    def collision_detected(self, other_particle) -> bool:
        return collision_detected(self.shapes, other_particle.shapes)

    def random_position_inside(self) -> Vector:
        shape = np.random.choice(self.shapes)
        return shape.random_position_inside()

    def closest_surface_position(self, position: Vector) -> Vector:
        surface_positions = [shape.closest_surface_position(position) for shape in self.shapes]
        return min(surface_positions, key = lambda surface_position : (surface_position - position).mag)
        #return surface_positions[np.argmin([(surface_position - position).mag for surface_position in surface_positions])]
        
    @abstractmethod
    def form_array(self, qx_array: np.ndarray, qy_array: np.ndarray, orientation: Vector) -> np.ndarray:
        pass

    @abstractmethod
    def magnetic_form_array(self, qx_array: np.ndarray, qy_array: np.ndarray, orientation: Vector, magnetization: Vector) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    def delta_sld(self, sld: float) -> float:
        return (sld - self.solvent_sld) * 1e-6

    def get_loggable_data(self) -> dict:
        return {
        'Particle type': type(self).__name__,
        **self.position.to_dict("Position"),
        **self.orientation.to_dict("Orientation"),
        **self.magnetization.to_dict("Magnetization"),
        'Volume' : self.volume,
        'Total scattering length' : self.scattering_length,
    }

@array_cache(max_size=500)
def form_array_sphere(radius: float, sld: float, q_array: np.ndarray) -> np.ndarray:
    volume = sphere_volume(radius)
    theta_arr = theta(q_array * radius)
    return sld * volume * theta_arr


if __name__ == "__main__":
    pass


# %%
