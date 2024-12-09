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
        ) for _ in range(box_number)]

def create_particle_iterator(particle_factory: ParticleFactory, nominal_concentration: float, dimensions: Sequence[float]) -> Iterator[ParticleResult]:
    current_volume = 0
    while nominal_concentration > (current_volume / create_cube(dimensions).get_volume()):
        particle_result = particle_factory.create_particle_result()
        current_volume = current_volume + particle_result.get_particle().get_volume()
        yield particle_result

def create_box_list_without_particle_number(box_number: int, nominal_concentration: float, particle_factory: ParticleFactory, dimensions: Sequence[float]) -> list[Box]:
    return [Box(
        particle_results = [particle for particle in create_particle_iterator(particle_factory=particle_factory, nominal_concentration=nominal_concentration, dimensions=dimensions)], 
        cube = create_cube(dimensions)) for _ in range(box_number)]

def create_box_iterator(particle_factory: ParticleFactory, particle_number: int, nominal_concentration: float, dimensions: Sequence[float]) -> Iterator[Box]:
    current_particle_numbers = 0
    while current_particle_numbers < particle_number:
        particles = [particle for particle in create_particle_iterator(particle_factory, nominal_concentration, dimensions)]
        current_particle_numbers = current_particle_numbers + len(particles)
        yield Box(particles, cube = create_cube(dimensions))

def create_box_list_without_box_number(particle_number: int, nominal_concentration: float, particle_factory: ParticleFactory, dimensions: Sequence[float]) -> list[Box]:
    return [box for box in create_box_iterator(particle_factory, particle_number, nominal_concentration, dimensions)]

def force_new_box(box: Box, box_creation_function: Callable[[Box, int], Box]) -> Box:
    l = len(box)
    for particle_index in range(l):
        for _ in range(100_000):
            box = box_creation_function(box, particle_index)
            if not box.wall_or_particle_collision(particle_index):
                break
        else:
            raise ValueError("Box is too dense to resolve")
    return box
        
def force_inside_box(box: Box) -> Box:
    return force_new_box(box, lambda b, p : b.move_inside_box(p))
    
def force_to_plane(box: Box) -> Box:
    return force_new_box(box, lambda b, p : b.move_to_plane(p))


@pydantic_dataclass
class BoxFactory:
    particle_number: int | None = None
    nominal_concentration: float | None = None
    box_number: int | None = None
    confine_to_plane: bool = True

    def organize_box_list(self, box_list: list[Box]) -> list[Box]:
        if self.confine_to_plane:
            return [force_to_plane(box) for box in box_list]
        return [force_inside_box(box) for box in box_list]
    
    def missing_particle_number(self, particle_factory: ParticleFactory, dimensions: Sequence[float]) -> list[Box]:
        if not self.nominal_concentration:
            raise TypeError("Nominal concentration is missing")
        if not self.box_number:
            raise TypeError("Box number is missing")
        box_list = create_box_list_without_particle_number(self.box_number, self.nominal_concentration, particle_factory, dimensions)
        return self.organize_box_list(box_list)
    
    def missing_box_number(self, particle_factory: ParticleFactory, dimensions: Sequence[float]) -> list[Box]:
        if not self.nominal_concentration:
            raise TypeError("Nominal concentration is missing")
        if not self.particle_number:
            raise TypeError("Particle number is missing")
        box_list = create_box_list_without_box_number(self.particle_number, self.nominal_concentration, particle_factory, dimensions)
        return self.organize_box_list(box_list)

    def create_box_list(self, particle_factory: ParticleFactory, dimensions: Sequence[float]) -> list[Box]:
        if not self.particle_number:
            return self.missing_particle_number(particle_factory, dimensions)
        if not self.box_number:
            return self.missing_box_number(particle_factory, dimensions)
        box_list = create_box_list_without_concentration(self.box_number, self.particle_number, particle_factory, dimensions)
        return self.organize_box_list(box_list)

    @classmethod
    def create_from_dataframes(cls, dataframes: dict[str, pd.DataFrame]):
        value_frame = parse_data.parse_value_frame(dataframes['Simulation parameters'])
        return cls(**value_frame)

if __name__ == "__main__":
    pass

