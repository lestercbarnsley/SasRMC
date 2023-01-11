#%%
from dataclasses import dataclass, field
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
class MoveByAndEnlargeCylinder(commands.ParticleCommand):
    change_by_factor: float
    position_delta: Vector

    def execute(self) -> None:
        EnlargeCylinder(self.box, self.particle_index, change_by_factor=self.change_by_factor).execute()
        commands.MoveParticleBy(self.box, self.particle_index, position_delta=self.position_delta).execute()
        
@dataclass
class LatticeCommandFactory(command_factory.CommandFactory):
    reference_angle: float

    def create_command(self, box: Box, particle_index: int, simulation_params: SimulationParams = None) -> commands.Command:
        return commands.FormLattice(box, particle_index, reference_particle_index=command_factory.different_random_int(len(box), particle_index), reference_angle=self.reference_angle)

@dataclass
class FixedJumpFactory(command_factory.CommandFactory):
    default_loc: float = 100.0
    default_scale: float = 10.0

    def create_command(self, box: Box, particle_index: int, simulation_params: SimulationParams = None) -> commands.Command:
        return commands.JumpParticleFixedDistance(box, particle_index, reference_particle_index=command_factory.different_random_int(len(box), particle_index), fixed_distance=np.random.normal(loc = self.default_loc, scale = self.default_scale ) )


'''@dataclass
class MoveMultipleParticles(commands.ParticleCommand):
    possible_commands: List[commands.ParticleCommand]

    def execute(self) -> None:
        for command in self.possible_commands:
            command.execute()


@dataclass
class MoveMultipleParticleFactory(command_factory.CommandFactory):
    command_number: int
    command_factory_list: List[command_factory.CommandFactory] = field(default_factory=list, repr = False, init = False)

    def set_command_list(self, command_factory_list: List[command_factory.CommandFactory]):
        self.command_factory_list = [command_factory_ for command_factory_ in command_factory_list] # make a new list

    def create_command(self, box: Box, particle_index: int, simulation_params: SimulationParams = None) -> commands.Command:
        possible_commands_factory = [np.random.choice(self.command_factory_list) for _ in range(self.command_number)]
        possible_commands = [possible_command.create_command(box, particle_index if i == 0 else np.random.choice(range(len(box.particles))), simulation_params) for i, possible_command in enumerate(possible_commands_factory)]
        return MoveMultipleParticles(box, particle_index, possible_commands = possible_commands)'''


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
            command_factory.JumpToParticleFactory(), 
            command_factory.NuclearMagneticRescaleFactory(change_by_factor=change_by_factor),
            command_factory.NuclearMagneticRescaleFactory(change_by_factor=massive_change_factor),
            EnlargeCylinderCommandFactory(change_by_factor**0.1),
            LatticeCommandFactory(massive_angle_change),
            FixedJumpFactory(),
            #EnlargeCylinderCommandFactory(massive_change_factor)
            ]
        #multiple_move_factory = MoveMultipleParticleFactory(np.random.choice(range(6)))
        #multiple_move_factory.set_command_list(command_list)
        #command_list.append(multiple_move_factory)
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
            radius=np.abs(polydisperse_parameter(loc = self.cylinder_radius, polyd= self.cylinder_radius_polydispersity)),
            height = np.abs(polydisperse_parameter(loc = self.cylinder_height, polyd=self.cylinder_height_polydispersity)),
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
            radius=np.abs(polydisperse_parameter(loc = self.cylinder_radius, polyd= self.cylinder_radius_polydispersity)),
            height = np.abs(polydisperse_parameter(loc = self.cylinder_height, polyd=self.cylinder_height_polydispersity)),
            cylinder_sld=self.cylinder_sld,
            solvent_sld=self.solvent_sld,
            position=Vector(0,0,0),
            orientation=Vector(0,0,1),
        )


if __name__ == "__main__":
    pass