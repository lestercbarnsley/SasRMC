#%%
from collections.abc import Callable
from dataclasses import dataclass

from typing_extensions import Self

from sas_rmc import Vector, constants
from sas_rmc.particles import ParticleResult, Particle
from sas_rmc.shapes import Cube


PI = constants.PI
rng = constants.RNG


def random_position_inside_cube(cube: Cube) -> Vector:
    return cube.random_position_inside()

def random_position_on_cube_plane(cube: Cube) -> Vector:
    return cube.random_position_inside().project_to_xy()


@dataclass
class Box:
    '''The Box class is responsible for particle mechanics, determining how and when particles can move. It has no knowledge about anything related to scattering, which is the responsibility of the Simulator class. Special cases of the Box class can be made as their own class that inherit from the Box'''
    particle_results: list[ParticleResult]
    cube: Cube

    #IF the box has no responsibility for the scattering, it should have no knowledge of the nuclear and magnetic rescale factors
    #Now that we're using the controller pattern, we can start getting rid of the "actions" in this class, because all actions are now the responsibility of Command subclasses

    def __len__(self) -> int:
        return len(self.particle_results)

    @property
    def volume(self) -> float:
        return self.cube.get_volume()
    
    def get_particles(self) -> list[Particle]:
        return [particle_res.get_particle() for particle_res in self.particle_results]

    def is_inside(self, position) -> bool:
        return self.cube.is_inside(position)
    
    def get_particle(self, particle_index: int) -> Particle:
        return self.particle_results[particle_index].get_particle()
    
    def wall_or_particle_collision(self, particle_index: int) -> bool:
        particle = self.get_particle(particle_index)
        if not self.is_inside(particle.get_position()):
            return True
        return any(particle.collision_detected(particle_j) for j, particle_j in enumerate(self.get_particles()) if j!=particle_index)

    def change_particle(self, particle_index: int, new_particle: Particle) -> Self:
        new_particle_res = self.particle_results[particle_index].change_particle(new_particle)
        return type(self)(
            particle_results=[particle_res if j!=particle_index else new_particle_res for j, particle_res in enumerate(self.particle_results)],
            cube = self.cube
        )
    
    def move_to_new_position(self, particle_index: int, new_position: Vector) -> Self:
        new_particle = self.get_particle(particle_index).change_position(new_position)
        return self.change_particle(particle_index, new_particle)

    def move_within_box(self, particle_index: int, cube_position_function: Callable[[Cube], Vector]) -> Self:
        new_position = cube_position_function(self.cube)
        return self.move_to_new_position(particle_index, new_position)
    
    def move_inside_box(self, particle_index: int) -> Self:
        return self.move_within_box(particle_index, random_position_inside_cube)
        
    def move_to_plane(self, particle_index: int) -> Self:
        return self.move_within_box(particle_index, random_position_on_cube_plane)
    
    def collision_test(self) -> bool:
        if not all(self.is_inside(particle.get_position()) for particle in self.get_particles()):
            return True
        return any(particle_i.collision_detected(particle_j) for i, particle_i in enumerate(self.get_particles()) for j, particle_j in enumerate(self.get_particles()) if i > j)
    
    def get_nearest_particle(self, particle_index: int) -> Particle:
        current_particle_position = self.get_particle(particle_index).get_position()
        def distance_from_particle(particle: Particle) -> float:
            return current_particle_position.distance_from_vector(particle.get_position())
        return min(
            [particle for i, particle in enumerate(self.get_particles()) if i != particle_index],
            key = distance_from_particle
            )
        
    def get_loggable_data(self) -> dict:
        return {
            f'Particle {i}' : particle_res.get_loggable_data() 
            for i, particle_res 
            in enumerate(self.particle_results)
        } | {
            'Dimension 0' : self.cube.dimension_0,
            'Dimension 1' : self.cube.dimension_1,
            'Dimension 2' : self.cube.dimension_2
        }

    
if __name__ == "__main__":
    pass

#%%