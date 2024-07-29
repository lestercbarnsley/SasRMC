#%%
from dataclasses import dataclass

from typing_extensions import Self

from sas_rmc import Vector, constants
from sas_rmc.particles import Particle
from sas_rmc.shapes import Cube


PI = constants.PI
rng = constants.RNG




@dataclass
class Box:
    '''The Box class is responsible for particle mechanics, determining how and when particles can move. It has no knowledge about anything related to scattering, which is the responsibility of the Simulator class. Special cases of the Box class can be made as their own class that inherit from the Box'''
    particles: list[Particle]
    cube: Cube

    #IF the box has no responsibility for the scattering, it should have no knowledge of the nuclear and magnetic rescale factors
    #Now that we're using the controller pattern, we can start getting rid of the "actions" in this class, because all actions are now the responsibility of Command subclasses

    def __len__(self) -> int:
        return len(self.particles)


    @property
    def volume(self) -> float:
        return self.cube.get_volume()

    def is_inside(self, position) -> bool:
        return self.cube.is_inside(position)
    
    def wall_or_particle_collision(self, i: int) -> bool: # Mark for deletion
        particle = self.particles[i]
        if not self.is_inside(particle.get_position()):
            return True
        return any(particle.collision_detected(particle_j) for particle_j in self.particles)
    
    def move_inside_box(self, i: int) -> Self:
        new_position = self.cube.random_position_inside()
        return type(self)(
            particles=[particle if j!=i else particle.change_position(new_position) for j, particle in enumerate(self.particles)],
            cube=self.cube
        )
    
    def collision_test(self) -> bool:
        if not all(self.is_inside(particle.get_position()) for particle in self.particles):
            return True
        return any(particle_i.collision_detected(particle_j) for i, particle_i in enumerate(self.particles) for j, particle_j in enumerate(self.particles) if i > j)
    
    def force_inside_box(self) -> Self:
        box = self
        l = len(box)
        for i in range(10_000_000):
            box = box.move_inside_box(i % l)
            if not box.collision_test():
                return box
        raise ValueError("Box is too dense to resolve")

    def get_nearest_particle(self, position: Vector) -> Particle:
        return min(self.particles, key = lambda particle : position.distance_from_vector(particle.get_position()))
        
    def get_loggable_data(self) -> dict:
        return {
            f'Particle {i}' : particle.get_loggable_data() 
            for i, particle 
            in enumerate(self.particles)
        } | {
            'Dimension 0' : self.cube.dimension_0,
            'Dimension 1' : self.cube.dimension_1,
            'Dimension 2' : self.cube.dimension_2
        }

    
if __name__ == "__main__":
    pass

#%%