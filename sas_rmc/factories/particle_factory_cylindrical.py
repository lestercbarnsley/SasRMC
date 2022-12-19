#%%
from dataclasses import dataclass
from typing import List

import numpy as np

from .particle_factory import ParticleFactory, polydisperse_parameter
from . import command_factory
from ..particles import CylindricalParticle, CylinderLong
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
        old_radius = old_particle.shapes[0].radius
        new_radius = self.change_by_factor * old_radius
        new_particle = old_particle.set_radius(new_radius)
        commands.SetParticleState(self.box, self.particle_index, new_particle).execute()


@dataclass
class EnlargeAllCylinders(commands.ParticleCommand):
    change_by_factor: float

    def execute(self) -> None:
        enlarge_cylinder_commands = [EnlargeCylinder(
            self.box, 
            particle_index, 
            self.change_by_factor) for particle_index, _ in enumerate(self.box.particles)]
        for enlarge_cylinder_command in enlarge_cylinder_commands:
            enlarge_cylinder_command.execute()


@dataclass
class EnlargeCylinderCommandFactory(command_factory.CommandFactory):
    change_by_factor: float

    def create_single_command(self, box: Box, particle_index: int, simulation_params: SimulationParams = None) -> EnlargeCylinder:
        return EnlargeCylinder(box, particle_index=particle_index, change_by_factor=self.change_by_factor)

    def create_all_command(self, box: Box, particle_index: int, simulation_params: SimulationParams = None) -> commands.Command:
        return EnlargeAllCylinders(box, particle_index, change_by_factor=self.change_by_factor)

    def create_command(self, box: Box, particle_index: int, simulation_params: SimulationParams = None) -> commands.Command:
        create = np.random.choice([self.create_single_command, self.create_all_command])
        #create = self.create_single_command
        return create(box, particle_index, simulation_params)


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
        actual_angle_change = rng.normal(loc = 0.0, scale = self.nominal_angle_change)
        massive_angle_change = rng.uniform(low = -PI, high = PI)
        command_list = [
            command_factory.MoveParticleToFactory(self.in_plane),
            command_factory.MoveParticleByFactory(position_delta),
            command_factory.OrbitParticleFactory(actual_angle_change=actual_angle_change),
            command_factory.OrbitParticleFactory(actual_angle_change=massive_angle_change),
            command_factory.NuclearMagneticRescaleFactory(change_by_factor=change_by_factor),
            command_factory.NuclearMagneticRescaleFactory(change_by_factor=massive_change_factor),
            #EnlargeCylinderCommandFactory(change_by_factor),
            #EnlargeCylinderCommandFactory(massive_change_factor)
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
            orientation=Vector(0,0,1),
        )

    @classmethod
    def gen_from_dict(cls, d: dict):
        pass

@dataclass
class CylindricalLongParticleFactory(CylindricalParticleFactory):

    def create_particle(self) -> CylinderLong:
        return CylinderLong.gen_from_parameters(
            radius=polydisperse_parameter(loc = self.cylinder_radius, polyd= self.cylinder_radius_polydispersity),
            height = polydisperse_parameter(loc = self.cylinder_height, polyd=self.cylinder_height_polydispersity),
            cylinder_sld=self.cylinder_sld,
            solvent_sld=self.solvent_sld,
            position=Vector(0,0,0),
            orientation=Vector(0,0,1),
        )


if __name__ == "__main__":
    pass