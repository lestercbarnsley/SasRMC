


from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
from typing_extensions import Self

from sas_rmc import Vector
from sas_rmc.particles import Particle, FormResult, ParticleResult
from sas_rmc.shapes import Shape


@dataclass
class ParticleArray(ParticleResult): # This is essentially an abstract wrapper class that is compatible with certain concrete result calculator implementations while still being compatible with the usual scattering simulation
    bound_particle: Particle

    def get_bound_particle(self) -> Particle:
        return self.bound_particle

    def get_particle(self) -> Particle:
        return self.get_bound_particle()
    
    def change_particle(self, particle: Particle) -> Self:
        return type(self)(bound_particle=particle)

    @abstractmethod
    def form_result(self, qx_array: np.ndarray, qy_array: np.ndarray) -> FormResult:
        pass

    def get_loggable_data(self) -> dict:
        loggable_data = super().get_loggable_data()
        return loggable_data | self.get_bound_particle().get_loggable_data()
        # I think the form particle wrapper should typically be transparent to the logger
