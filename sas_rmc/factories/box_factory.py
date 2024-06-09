#%%
from collections.abc import Callable
from typing import Iterator


from sas_rmc import Particle, Vector, Box
from sas_rmc.shapes import Cube

def create_cube(dimensions: tuple[float]) -> Cube:
    return Cube(
        central_position=Vector.null_vector(),
        orientation=Vector(0, 1, 0),
        dimension_0=dimensions[0],
        dimension_1=dimensions[1], 
        dimension_2=dimensions[2]
    )

def create_box(box_number: int, particle_number: int, particle_factory: Callable[[], Particle], dimensions: tuple[float]):
    return [Box(
        particles = [particle_factory() for _ in range(particle_number)], 
        cube=create_cube(dimensions),
        ) for _ in range(box_number)]

def create_particle_iterator(particle_factory: Callable[[], Particle], nominal_concentration: float, dimensions: tuple[float, float, float]) -> Iterator[Particle]:
    current_volume = 0
    while nominal_concentration > (current_volume / create_cube(dimensions).get_volume()):
        particle = particle_factory()
        current_volume = current_volume + particle.get_volume()
        yield particle

def create_box_2(box_number: int, nominal_concentration: float, particle_factory: Callable[[], Particle], dimensions: tuple[float]) -> list[Box]:
    return [Box(
        particles = [particle for particle in create_particle_iterator(particle_factory=particle_factory, nominal_concentration=nominal_concentration, dimensions=dimensions)], 
        cube = create_cube(dimensions)) for _ in box_number]

def create_box_iterator(particle_factory: Callable[[], Particle], particle_number: int, nominal_concentration: float, dimensions: tuple[float]) -> Iterator[Box]:
    current_particle_numbers = 0
    while current_particle_numbers < particle_number:
        particles = [particle for particle in create_particle_iterator(particle_factory,nominal_concentration, dimensions)]
        current_particle_numbers = current_particle_numbers + len(particles)
        yield Box(particles, cube = create_cube(dimensions))

def create_box_3(particle_number: int, nominal_concentration: float, particle_factory: Callable[[], Particle], dimensions: tuple[float]) -> list[Box]:
    return [box for box in create_box_iterator(particle_factory, particle_number, nominal_concentration, dimensions)]



def gen_list_factory_from_dict(d: dict) -> list[Box]:
    nominal_concentration = d.get("nominal_concentration", 0.0)
    particle_number = d.get("particle_number", 0)
    box_number = d.get("box_number", 0)
    
    if particle_number == 0:
        return create_box(box_number=box_number, nominal_concentration=nominal_concentration)
    if box_number == 0:
        return BoxListParticleNumberConcentration(particle_number=particle_number, nominal_concentration=nominal_concentration)
    return BoxListParticleNumberBoxNumber(particle_number=particle_number, box_number=box_number)


if __name__ == "__main__":
    pass

