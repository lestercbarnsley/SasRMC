from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from ..vector import Vector
from .. import commands
from ..box_simulation import Box
from ..scattering_simulation import SimulationParams
from .. import constants

rng = constants.RNG


def different_random_int(n: int, number_to_avoid: int) -> int:
    for _ in range(200000):
        x = rng.choice(range(n))
        if x != number_to_avoid:
            return x
    return -1


@dataclass
class CommandFactory(ABC):
    
    @abstractmethod
    def create_command(self, box: Box, particle_index: int, simulation_params: SimulationParams = None) -> commands.Command:
        pass


@dataclass  
class CommandFactoryList(CommandFactory):
    
    @abstractmethod # This should be abstract because not every factory list is also a factory list
    def create_command_list(self) -> List[CommandFactory]:
        pass

    def create_command(self, box: Box, particle_index: int, simulation_params: SimulationParams = None) -> commands.Command:
        chosen_command = rng.choice(self.create_command_list())
        return chosen_command.create_command(box, particle_index, simulation_params)


@dataclass
class MoveParticleByFactory(CommandFactory):
    position_delta: Vector

    def create_command(self, box: Box, particle_index: int, simulation_params: SimulationParams = None) -> commands.MoveParticleBy:
        return commands.MoveParticleBy(
            box = box,
            particle_index=particle_index,
            position_delta=self.position_delta)
        

@dataclass
class JumpToParticleFactory(CommandFactory):

    def create_command(self, box: Box, particle_index: int, simulation_params: SimulationParams = None) -> commands.JumpParticleTo:
        return commands.JumpParticleTo(
            box = box,
            particle_index=particle_index,
            reference_particle_index=different_random_int(len(box), particle_index))


@dataclass
class OrbitParticleFactory(CommandFactory):
    actual_angle_change: float

    def create_command(self, box: Box, particle_index: int, simulation_params: SimulationParams = None) -> commands.OrbitParticle:
        return commands.OrbitParticle(
            box=box,
            particle_index=particle_index,
            relative_angle=self.actual_angle_change)


@dataclass
class MoveParticleToFactory(CommandFactory):
    in_plane: bool

    def create_command(self, box: Box, particle_index: int, simulation_params: SimulationParams = None) -> commands.MoveParticleTo:
        position_new = box.cube.random_position_inside()
        if self.in_plane:
            position_new = Vector(position_new.x, position_new.y)
        return commands.MoveParticleTo(
            box = box,
            particle_index=particle_index,
            position_new=position_new)


@dataclass
class RotateParticleFactory(CommandFactory):
    actual_angle_change: float

    def create_command(self, box: Box, particle_index: int, simulation_params: SimulationParams = None) -> commands.RotateParticle:
        return commands.RotateParticle(
            box = box,
            particle_index=particle_index,
            relative_angle=self.actual_angle_change)


@dataclass
class RotateMagnetizationFactory(CommandFactory):
    actual_angle_change: float

    def create_command(self, box: Box, particle_index: int, simulation_params: SimulationParams = None) -> commands.RotateMagnetization:
        return commands.RotateMagnetization(
            box = box,
            particle_index= particle_index,
            relative_angle=self.actual_angle_change)


@dataclass
class FlipMagnetizationFactory(CommandFactory):

    def create_command(self, box: Box, particle_index: int, simulation_params: SimulationParams = None) -> commands.FlipMagnetization:
        return commands.FlipMagnetization(
            box = box,
            particle_index= particle_index)


@dataclass
class NuclearMagneticRescaleFactory(CommandFactory):
    change_by_factor: float

    def create_command(self, box: Box, particle_index: int, simulation_params: SimulationParams = None) -> commands.NuclearMagneticRescale:
        return commands.NuclearMagneticRescale(
            simulation_params=simulation_params,
            change_by_factor=self.change_by_factor
            )


@dataclass
class CompressShellFactory(CommandFactory):
    change_by_factor: float

    def create_all_command(self, box: Box, particle_index: int, simulation_params: SimulationParams = None) -> commands.CompressAllShells:
        return commands.CompressAllShells(
            box = box,
            particle_index=particle_index,
            change_by_factor=self.change_by_factor
        )

    def create_single_command(self, box: Box, particle_index: int, simulation_params: SimulationParams = None) -> commands.CompressShell:
        return commands.CompressShell(
            box=box,
            particle_index=particle_index,
            change_by_factor=self.change_by_factor,
            reference_particle_index=different_random_int(len(box), particle_index),
        )

    def create_command(self, box: Box, particle_index: int, simulation_params: SimulationParams = None) -> commands.Command:
        cmd = rng.choice([self.create_all_command, self.create_single_command])
        return cmd(box, particle_index, simulation_params)

