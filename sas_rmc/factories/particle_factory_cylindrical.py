#%%
from dataclasses import dataclass
from typing import List

import numpy as np

from .particle_factory import ParticleFactory, polydisperse_parameter
from . import command_factory
from ..particles import CylindricalParticle
from ..box_simulation import Box
from .. scattering_simulation import SimulationParams
from .. import Vector, constants, commands

rng = constants.RNG
PI = constants.PI


@dataclass
class EnlargeCylinder(commands.ParticleCommand):
    change_by_factor: float

    def execute(self) -> None:
        old_particle: CylindricalParticle = self.particle
        cylinder_shape = old_particle.shapes[0]
        new_particle = CylindricalParticle.gen_from_parameters(
            radius = self.change_by_factor * (cylinder_shape.radius),
            height=cylinder_shape.height,
            cylinder_sld=old_particle.cylinder_sld,
            solvent_sld=old_particle.solvent_sld,
            position=old_particle.position,
            orientation=old_particle.orientation,
        )
        commands.SetParticleState(self.box, self.particle_index, new_particle).execute()


@dataclass
class EnlargeCylinderCommandFactory(command_factory.CommandFactory):
    change_by_factor: float

    def create_command(self, box: Box, particle_index: int, simulation_params: SimulationParams = None) -> commands.Command:
        return EnlargeCylinder(box, particle_index=particle_index, change_by_factor=self.change_by_factor)


@dataclass
class CylindricalCommandFactory(command_factory.CommandFactoryList):
    in_plane: bool = True
    nominal_step_size: float = 100
    nominal_angle_change: float = PI/8
    nominal_rescale_change: float = 0.02

    def create_command_list(self) -> List[command_factory.CommandFactory]:
        position_delta_size = rng.normal(loc = 0.0, scale = self.nominal_step_size)
        position_delta = (Vector.random_vector_xy if self.in_plane else Vector.random_vector)(position_delta_size)
        change_by_factor = np.abs(rng.normal(loc = 1.0, scale = self.nominal_rescale_change))
        massive_change_factor = change_by_factor**10
        command_list = [
            command_factory.MoveParticleToFactory(self.in_plane),
            command_factory.MoveParticleByFactory(position_delta),
            command_factory.NuclearMagneticRescaleFactory(change_by_factor=change_by_factor),
            command_factory.NuclearMagneticRescaleFactory(change_by_factor=massive_change_factor),
            EnlargeCylinderCommandFactory(change_by_factor),
            EnlargeCylinder(massive_change_factor)
            ]
        return command_list


@dataclass
class CylindricalParticleFactory(ParticleFactory):
    command_factory: CylindricalCommandFactory
    cylinder_radius: float
    cylinder_radius_polydispersity: float
    cylinder_height: float
    cylinder_height_polydispersity: float
    cylinder_sld: float
    solvent_sld: float

    def create_particle(self) -> CylindricalParticle:
        return CylindricalParticle.gen_from_parameters(
            radius=polydisperse_parameter(loc = self.cylinder_radius, polyd= self.cylinder_radius_polydispersity),
            height = polydisperse_parameter(loc = self.cylinder_height, polyd=self.cylinder_height_polydispersity),
            cylinder_sld=self.cylinder_sld,
            solvent_sld=self.solvent_sld,
            position=Vector(0,0,0),
            orientation=Vector(0,1,0),
        )



if __name__ == "__main__":
    pass