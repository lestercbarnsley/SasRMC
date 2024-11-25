


from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
from typing_extensions import Self

from sas_rmc import Vector
from sas_rmc.particles import Particle, FormResult
from sas_rmc.shapes import Shape


@dataclass
class ParticleArray(Particle): # This is essentially an abstract wrapper class that is compatible with certain concrete result calculator implementations while still being compatible with the usual scattering simulation

    @abstractmethod
    def get_bound_particle(self) -> Particle:
        pass

    @abstractmethod
    def change_bound_particle(self, bound_particle: Particle) -> Self:
        pass

    def get_position(self) -> Vector:
        return self.get_bound_particle().get_position()
    
    def get_orientation(self) -> Vector:
        return self.get_bound_particle().get_orientation()
    
    def get_magnetization(self) -> Vector:
        return self.get_bound_particle().get_magnetization()
    
    def get_volume(self) -> float:
        return self.get_bound_particle().get_volume()
    
    def get_shapes(self) -> list[Shape]:
        return self.get_bound_particle().get_shapes()
    
    def is_inside(self, position: Vector) -> bool:
        return self.get_bound_particle().is_inside(position)
    
    def collision_detected(self, other_particle: Self) -> bool:
        return self.get_bound_particle().collision_detected(other_particle)
    
    def get_scattering_length(self) -> float:
        return self.get_bound_particle().get_scattering_length()
    
    def is_magnetic(self) -> bool:
        return self.get_bound_particle().is_magnetic()
    
    def change_position(self, position: Vector) -> Self:
        new_bound_particle = self.get_bound_particle().change_position(position)
        return self.change_bound_particle(new_bound_particle)
    
    def change_orientation(self, orientation: Vector) -> Self:
        new_bound_particle = self.get_bound_particle().change_orientation(orientation)
        return self.change_bound_particle(new_bound_particle)
    
    def change_magnetization(self, magnetization: Vector) -> Self:
        new_bound_particle = self.get_bound_particle().change_magnetization(magnetization)
        return self.change_bound_particle(new_bound_particle)
    
    def change_particle(self, particle: Self) -> Self:
        return self.change_bound_particle(particle)

    @abstractmethod
    def form_result(self, qx_array: np.ndarray, qy_array: np.ndarray) -> FormResult:
        pass

    def get_loggable_data(self) -> dict:
        loggable_data = super().get_loggable_data()
        return loggable_data | self.get_bound_particle().get_loggable_data()
        # I think the form particle wrapper should typically be transparent to the logger
