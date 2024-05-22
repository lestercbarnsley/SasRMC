#%%
from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np

from sas_rmc.shapes import Shape, collision_detected
from sas_rmc import Vector, constants


PI = constants.PI
GAMMA_N = constants.GAMMA_N
R_0 = constants.R_0
BOHR_MAG = constants.BOHR_MAG
B_H_IN_INVERSE_AMP_METRES = constants.B_H_IN_INVERSE_AMP_METRES


def magnetic_sld_in_angstrom_minus_2(magnetization_vector_in_amp_per_metre: Vector) -> tuple[float, float, float]:
    # Let us do all calculations in metres, then convert to Angstrom^-2 as the last step
    magnetization = magnetization_vector_in_amp_per_metre
    sld_vector = B_H_IN_INVERSE_AMP_METRES * magnetization / (1e10**2)
    return sld_vector.x, sld_vector.y, sld_vector.z


@dataclass
class FormResult:
    form_nuclear: np.ndarray
    form_magnetic_x: np.ndarray
    form_magnetic_y: np.ndarray
    form_magnetic_z: np.ndarray


@dataclass
class Particle(ABC):
    """Abstract base class for Particle type.

    When you make your own particle type, you should inherit from this class. This is an abstract class, so it cannot be instantiated itself, but it shows a template of how a Particle class should be written, and provides abstract methods that should have implementations written by the user.

    
    """

    @abstractmethod
    def get_position(self) -> Vector:
        pass

    @abstractmethod
    def get_orientation(self) -> Vector:
        pass

    @abstractmethod
    def get_magnetization(self) -> Vector:
        pass

    @abstractmethod
    def get_volume(self) -> float:
        pass

    @abstractmethod
    def form_result(self, qx_array: np.ndarray, qy_array: np.ndarray) -> FormResult:
        pass

    @abstractmethod
    def get_shapes(self) -> list[Shape]:
        pass

    def is_inside(self, position: Vector) -> bool:
        return any(shape.is_inside(position) for shape in self.get_shapes())

    def collision_detected(self, other_particle: Particle) -> bool:
        return collision_detected(self.get_shapes(), other_particle.get_shapes())

    @abstractmethod
    def get_scattering_length(self) -> float:
        pass

    def is_magnetic(self) -> bool:
        return self.get_magnetization().mag != 0

    @abstractmethod
    def change_position(self, position: Vector) -> Particle:
        pass

    @abstractmethod
    def change_orientation(self, orientation: Vector) -> Particle:
        pass

    @abstractmethod
    def change_magnetization(self, magnetization: Vector) -> Particle:
        pass
    
    @abstractmethod
    def get_loggable_data(self) -> dict: # This is a template, but it must be overrided
        return {
            'Particle type': type(self).__name__,
            **self.get_position().to_dict("Position"),
            **self.get_orientation().to_dict("Orientation"),
            **self.get_magnetization().to_dict("Magnetization"),
            'Volume' : self.get_volume(),
            'Total scattering length' : self.get_scattering_length(),
        }



if __name__ == "__main__":
    pass


# %%
