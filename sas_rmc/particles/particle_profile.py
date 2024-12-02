


from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
from typing_extensions import Self

from sas_rmc.particles import ParticleResult
from sas_rmc.particles.particle import Particle


@dataclass
class ParticleProfile(ParticleResult): # This is essentially an abstract wrapper class that is compatible with certain concrete result calculator implementations while still being compatible with the usual scattering simulation
    bound_particle: Particle

    def get_bound_particle(self) -> Particle:
        return self.bound_particle

    def get_particle(self) -> Particle:
        return self.get_bound_particle()
    
    def change_particle(self, particle: Particle) -> Self:
        return type(self)(bound_particle=particle)

    @abstractmethod
    def form_profile(self, q_profile: np.ndarray) -> np.ndarray:
        pass

    def get_loggable_data(self) -> dict:
        loggable_data = super().get_loggable_data()
        return loggable_data | self.get_bound_particle().get_loggable_data()
        # I think the form particle wrapper should typically be transparent to the logger

        
