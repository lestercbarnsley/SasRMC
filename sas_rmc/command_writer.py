#%%
from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol

import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import pyplot as plt

from .box_simulation import Box
from .particle import CoreShellParticle, Dumbbell, Particle

PI = np.pi

class Loggable(Protocol):
    def get_loggable_data(self) -> dict:
        pass


Converter = Callable[[Loggable], dict]
ParticleConverter = Callable[[Particle], dict]


def standard_command_converter(loggable: Loggable) -> dict:
    return loggable.get_loggable_data()


@dataclass
class CommandWriter:

    command_converter: Converter

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

def convert_core_shell_particle(core_shell_particle: CoreShellParticle) -> dict:
    core_radius = core_shell_particle.shapes[0].radius
    overall_radius = core_shell_particle.shapes[1].radius
    thickness = overall_radius - core_radius
    data = {
        'Particle type': "",
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

def put_particle_on_ax(particle: Particle, ax: Axes, c: str = 'blue', alpha: float = 0.5) -> None:
    if isinstance(particle, CoreShellParticle):
        circle = plt.Circle((particle.position.x, particle.position.y), radius= particle.shapes[1].radius, color = c, alpha = alpha)
        ax.add_patch(circle)
    elif isinstance(particle, Dumbbell):
        put_particle_on_ax(particle=particle.particle_1, ax = ax, c = 'blue')
        put_particle_on_ax(particle=particle.particle_2, ax = ax, c = 'red')
    else:
        ax.scatter(x = particle.position.x, y = particle.position.y, c = 'blue')

@dataclass
class BoxWriter:
    particle_converter: ParticleConverter
    
    def to_data(self, box: Box) -> List[dict]:
        return [self.particle_converter(particle) for particle in box.particles]

    def to_plot(self, box: Box, fontsize: int = 14, particle_on_axes_writer: Optional[Callable[[Particle, Axes], None]] = None) -> Figure:
        p_on_ax_writer = particle_on_axes_writer if particle_on_axes_writer is not None else put_particle_on_ax
        fig, ax = plt.subplots()
        fig.set_size_inches(4,4)
        for p in box.particles:
            p_on_ax_writer(p, ax)
        d_0, d_1 = box.cube.dimension_0, box.cube.dimension_1
        ax.set_xlim(-d_0 / 2, +d_0 / 2)
        ax.set_ylim(-d_1 / 2, +d_1 / 2)
        ax.set_xlabel(r'X (Angstrom)',fontsize =  fontsize)
        ax.set_ylabel(r'Y (Angstrom)',fontsize =  fontsize)

        ax.set_box_aspect(d_1 / d_0)
        fig.tight_layout()
        return fig

    @classmethod
    def standard_box_writer(cls):
        return cls(
            particle_converter = convert_particle,
        )


if __name__ == "__main__":
    pass

#%%
