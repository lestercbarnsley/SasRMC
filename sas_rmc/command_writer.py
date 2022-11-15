#%%
from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import pyplot as plt

from .converters import particle_to_axes
from .box_simulation import Box
from .particles.particle import Particle
from . import constants

PI = constants.PI


class Loggable(Protocol):
    def get_loggable_data(self) -> dict:
        pass


Converter = Callable[[Loggable], dict]
ParticleConverter = Callable[[Particle], dict]
ParticleToAxesWriter = Callable[[Particle, Axes], None]


@dataclass
class CommandWriter:

    command_converter: Converter

    def to_data(self, command: Loggable) -> dict:
        return self.command_converter(command)
        #return self.command_converter.convert(command)

    @classmethod
    def standard_particle_writer(cls):
        return cls(command_converter = lambda loggable: loggable.get_loggable_data())


@dataclass
class BoxWriter:
    particle_converter: ParticleConverter
    
    def to_data(self, box: Box) -> List[dict]:
        return [self.particle_converter(particle) for particle in box.particles]

    def to_plot(self, box: Box, fontsize: int = 14, particle_to_axes_writer: Optional[ParticleToAxesWriter] = None) -> Figure:
        p_on_ax_writer = particle_to_axes_writer if particle_to_axes_writer is not None else particle_to_axes
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
            particle_converter = lambda particle : particle.get_loggable_data(),
        )


if __name__ == "__main__":
    pass

#%%
