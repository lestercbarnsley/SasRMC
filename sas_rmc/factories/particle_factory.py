#%%

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .command_factory import CommandFactory
from ..box_simulation import Box
from ..particles import Particle
from ..commands import Command
from ..scattering_simulation import SimulationParams


rng = np.random.default_rng()


def polydisperse_parameter(loc: float, polyd: float, dispersity_fn: Callable[[float, float], float] = None) -> float:
    poly_fn = dispersity_fn if dispersity_fn else (lambda l, s: rng.normal(loc = l, scale = s)) # I only want to write out the lambda expression like this so I can be explicit about the kwargs
    return poly_fn(loc, loc * polyd)
    

@dataclass
class ParticleFactory(ABC):
    command_factory: CommandFactory

    @abstractmethod
    def create_particle(self) -> Particle:
        pass

    def calculate_effective_volume(self, particle_test_number: int = 100000) -> float:
        return np.average([self.create_particle().volume for _ in range(particle_test_number)])

    def create_command(self, box: Box, particle_index: int, simulation_params: SimulationParams = None) -> Command:
        return self.command_factory.create_command(box, particle_index, simulation_params)

    @classmethod
    @abstractmethod
    def gen_from_dict(cls, d: dict):
        pass
     


#%%
