#%%
from collections.abc import Callable
from typing import Iterator, Sequence


from sas_rmc import Particle, Vector
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

def create_box_list_without_concentration(box_number: int, particle_number: int, particle_factory: Callable[[], Particle], dimensions: Sequence[float]):
    return [Box(
        particles = [particle_factory() for _ in range(particle_number)], 
        cube=create_cube(dimensions),
        ).force_to_plane() for _ in range(box_number)]

def create_particle_iterator(particle_factory: Callable[[], Particle], nominal_concentration: float, dimensions: Sequence[float]) -> Iterator[Particle]:
    current_volume = 0
    while nominal_concentration > (current_volume / create_cube(dimensions).get_volume()):
        particle = particle_factory()
        current_volume = current_volume + particle.get_volume()
        yield particle

def create_box_list_without_particle_number(box_number: int, nominal_concentration: float, particle_factory: Callable[[], Particle], dimensions: Sequence[float]) -> list[Box]:
    return [Box(
        particles = [particle for particle in create_particle_iterator(particle_factory=particle_factory, nominal_concentration=nominal_concentration, dimensions=dimensions)], 
        cube = create_cube(dimensions)).force_to_plane() for _ in range(box_number)]

def create_box_iterator(particle_factory: Callable[[], Particle], particle_number: int, nominal_concentration: float, dimensions: Sequence[float]) -> Iterator[Box]:
    current_particle_numbers = 0
    while current_particle_numbers < particle_number:
        particles = [particle for particle in create_particle_iterator(particle_factory,nominal_concentration, dimensions)]
        current_particle_numbers = current_particle_numbers + len(particles)
        yield Box(particles, cube = create_cube(dimensions)).force_to_plane()
        

def create_box_list_without_box_number(particle_number: int, nominal_concentration: float, particle_factory: Callable[[], Particle], dimensions: Sequence[float]) -> list[Box]:
    return [box.force_to_plane() for box in create_box_iterator(particle_factory, particle_number, nominal_concentration, dimensions)]

def create_box_list(particle_factory: Callable[[], Particle], dimensions: Sequence[float], particle_number: int | None = None, box_number: int | None = None, nominal_concentration: float | None = None) -> list[Box]:
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

if __name__ == "__main__":
    print(Vector(0,1))

