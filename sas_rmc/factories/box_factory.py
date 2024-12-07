#%%
from collections.abc import Callable
from typing import Iterator, Sequence

import pandas as pd
from pydantic.dataclasses import dataclass as pydantic_dataclass

from sas_rmc import Vector
from sas_rmc.factories import parse_data
from sas_rmc.factories.particle_factory import ParticleFactory
from sas_rmc.particles import ParticleResult
from sas_rmc.box_simulation import Box
from sas_rmc.shapes import Cube


def create_cube(dimensions: Sequence[float]) -> Cube:
    return Cube(
        central_position=Vector.null_vector(),
        orientation=Vector(0, 1, 0),
        dimension_0=dimensions[0],
        dimension_1=dimensions[1], 
        dimension_2=dimensions[2]
    )

def create_box_list_without_concentration(box_number: int, particle_number: int, particle_factory: ParticleFactory, dimensions: Sequence[float]):
    return [Box(
        particle_results = [particle_factory.create_particle_result() for _ in range(int(particle_number / box_number))], 
        cube=create_cube(dimensions),
        ).force_to_plane() for _ in range(box_number)]

def create_particle_iterator(particle_factory: ParticleFactory, nominal_concentration: float, dimensions: Sequence[float]) -> Iterator[ParticleResult]:
    current_volume = 0
    while nominal_concentration > (current_volume / create_cube(dimensions).get_volume()):
        particle_result = particle_factory.create_particle_result()
        current_volume = current_volume + particle_result.get_particle().get_volume()
        yield particle_result

def create_box_list_without_particle_number(box_number: int, nominal_concentration: float, particle_factory: ParticleFactory, dimensions: Sequence[float]) -> list[Box]:
    return [Box(
        particle_results = [particle for particle in create_particle_iterator(particle_factory=particle_factory, nominal_concentration=nominal_concentration, dimensions=dimensions)], 
        cube = create_cube(dimensions)).force_to_plane() for _ in range(box_number)]

def create_box_iterator(particle_factory: ParticleFactory, particle_number: int, nominal_concentration: float, dimensions: Sequence[float]) -> Iterator[Box]:
    current_particle_numbers = 0
    while current_particle_numbers < particle_number:
        particles = [particle for particle in create_particle_iterator(particle_factory,nominal_concentration, dimensions)]
        current_particle_numbers = current_particle_numbers + len(particles)
        yield Box(particles, cube = create_cube(dimensions)).force_to_plane()
        

def create_box_list_without_box_number(particle_number: int, nominal_concentration: float, particle_factory: ParticleFactory, dimensions: Sequence[float]) -> list[Box]:
    return [box.force_to_plane() for box in create_box_iterator(particle_factory, particle_number, nominal_concentration, dimensions)]

def create_box_list(particle_factory: ParticleFactory, dimensions: Sequence[float], particle_number: int | None = None, box_number: int | None = None, nominal_concentration: float | None = None) -> list[Box]:
    if not particle_number:
        if not nominal_concentration:
            raise TypeError("Nominal concentration is missing")
        if not box_number:
            raise TypeError("Box number is missing")
        return create_box_list_without_particle_number(box_number, nominal_concentration, particle_factory, dimensions)
    if not box_number:
        if not nominal_concentration:
            raise TypeError("Nominal concentration is missing")
        if not particle_number:
            raise TypeError("Particle number is missing")
        return create_box_list_without_box_number(particle_number, nominal_concentration, particle_factory, dimensions)
    return create_box_list_without_concentration(box_number, particle_number, particle_factory, dimensions)


@pydantic_dataclass
class BoxFactory:
    particle_number: int | None = None
    nominal_concentration: float | None = None
    box_number: int | None = None

    def create_box_list(self, particle_factory: ParticleFactory, dimensions: Sequence[float]) -> list[Box]:
        if not self.particle_number:
            if not self.nominal_concentration:
                raise TypeError("Nominal concentration is missing")
            if not self.box_number:
                raise TypeError("Box number is missing")
            return create_box_list_without_particle_number(self.box_number, self.nominal_concentration, particle_factory, dimensions)
        if not self.box_number:
            if not self.nominal_concentration:
                raise TypeError("Nominal concentration is missing")
            if not self.particle_number:
                raise TypeError("Particle number is missing")
            return create_box_list_without_box_number(self.particle_number, self.nominal_concentration, particle_factory, dimensions)
        return create_box_list_without_concentration(self.box_number, self.particle_number, particle_factory, dimensions)

    @classmethod
    def create_from_dataframes(cls, dataframes: dict[str, pd.DataFrame]):
        value_frame = parse_data.parse_value_frame(dataframes['Simulation parameters'])
        return cls(**value_frame)

if __name__ == "__main__":
    pass

