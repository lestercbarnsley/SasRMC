from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Protocol

from .box_simulation import Box
from .particle import CoreShellParticle, Particle


class Loggable(Protocol):
    def get_loggable_data(self) -> dict:
        pass
    
    
@dataclass# Mark for deletion
class Converter(ABC):

    @abstractmethod
    def convert(self, command: Loggable) -> dict:
        pass


@dataclass# Mark for deletion
class ParticleConverter(Converter):
    conversion_function: Callable[[dict], dict] = None

    def convert(self, command: Loggable) -> dict:
        data = command.get_loggable_data()
        if self.conversion_function is not None:
            return self.conversion_function(data)
        return data


def standard_command_converter(loggable: Loggable) -> dict:
    return loggable.get_loggable_data()


@dataclass
class CommandWriter:#(ABC):

    '''@abstractmethod
    def to_data(self, command: Loggable) -> dict:
        pass

    



@dataclass
class ParticleWriter(CommandWriter):'''

    command_converter: Callable[[Loggable], dict]#Converter

    def to_data(self, command: Loggable) -> dict:
        return self.command_converter(command)
        #return self.command_converter.convert(command)

    @classmethod
    def standard_particle_writer(cls):
        return cls(command_converter = standard_command_converter)


def convert_default_particle(particle: Particle) -> dict:
    return {
        'Particle type': type(particle).__name__,
        'Position.X' : particle.position.x,
        'Position.Y' : particle.position.y,
        'Position.Z' : particle.position.z,
        'Orientation.X' : particle.orientation.x,
        'Orientation.Y' : particle.orientation.y,
        'Orientation.Z' : particle.orientation.z,
        'Magnetization.X' : particle.magnetization.x,
        'Magnetization.Y' : particle.magnetization.y,
        'Magnetization.Z' : particle.magnetization.z,
        'Volume' : particle.volume,
        'Total scattering length' : particle.scattering_length,
    }


def convert_core_shell_particle(core_shell_particle: CoreShellParticle):
    core_radius = core_shell_particle.shapes[0].radius
    overall_radius = core_shell_particle.shapes[1].radius
    thickness = overall_radius - core_radius
    data = {
        'Particle type': "this is about to be overwritten, so it doesn't matter what I write here",
        'Core radius': core_radius,
        'Shell thickness': thickness,
        'Core SLD': core_shell_particle.core_sld,
        'Shell SLD' : core_shell_particle.shell_sld,
        'Solvent SLD': core_shell_particle.solvent_sld,
    }
    data.update(
        convert_default_particle(core_shell_particle)
    )
    return data

def convert_particle(particle: Particle) -> dict:
    #When making these converters, ask the question, what parameters do I need to reconstruct a particle from a finished simulation
    if isinstance(particle, CoreShellParticle):
        return convert_core_shell_particle(particle)
    return convert_default_particle(particle)


@dataclass
class BoxWriter:
    particle_converter: Callable[[Particle], dict]

    def to_data(self, box: Box) -> List[dict]:
        return [self.particle_converter(particle) for particle in box.particles]

    @classmethod
    def standard_box_writer(cls):
        return cls(
            particle_converter = convert_particle
        )


if __name__ == "__main__":
    pass
