#%%

from dataclasses import dataclass, field
from typing import List, Callable

import numpy as np
from matplotlib import pyplot as plt

from .particles.particle import Particle
from .vector import Vector
from .shapes.shapes import Cube
from . import constants


PI = constants.PI
rng = constants.RNG


def collision_detected_3d(particle_1: Particle, particle_2: Particle) -> bool:
    if particle_1.collision_detected(particle_2):
        return True
    if particle_1.position.z == particle_2.position.z:
        return False
    test_position = Vector(particle_2.position.x, particle_2.position.y, particle_1.position.z) # The central position of the second particle can't be under the shadow of the first, but this doesn't mean that the two shadows can't overlap
    return particle_1.is_inside(test_position)


@dataclass
class Box:
    '''The Box class is responsible for particle mechanics, determining how and when particles can move. It has no knowledge about anything related to scattering, which is the responsibility of the Simulator class. Special cases of the Box class can be made as their own class that inherit from the Box'''
    particles: List[Particle] = field(default_factory=list)
    cube: Cube = field(default_factory=Cube)

    #IF the box has no responsibility for the scattering, it should have no knowledge of the nuclear and magnetic rescale factors
    #Now that we're using the controller pattern, we can start getting rid of the "actions" in this class, because all actions are now the responsibility of Command subclasses

    def __len__(self) -> int:
        return len(self.particles)

    def __getitem__(self, i: int) -> Particle:
        return self.particles[i]

    @property
    def volume(self) -> float:
        return self.cube.volume

    @property
    def n_iterations(self) -> int:
        return len(self.particles)

    @property
    def sizes(self) -> List[float]:
        cube = self.cube
        return [cube.dimension_0, cube.dimension_1, cube.dimension_2]

    def is_inside(self, position) -> bool:
        return self.cube.is_inside(position)

    def is_magnetic(self) -> bool:
        return any(particle.is_magnetic() for particle in self.particles)
    
    def wall_or_particle_collision(self, i: int, half_test = False, collision_detected_fn: Callable[[Particle, Particle], bool] = None) -> bool:
        particle = self.particles[i]
        if not self.is_inside(particle.position):
            return True
        compare_to_i = (lambda j: i < j) if half_test else (lambda j: i != j) #There's no way to get around this step, it will be done one way or another
        collision_detected_fn = collision_detected_fn if collision_detected_fn is not None else collision_detected_3d
        return any(compare_to_i(j) and collision_detected_fn(particle, particle2) for j, particle2 in enumerate(self.particles))
        

    def move_inside_box(self, i: int, in_plane: bool = False) -> None:
        particle = self.particles[i]
        position = self.cube.random_position_inside()
        position_set_inplane = lambda : particle.set_position(Vector(position.x, position.y, z = 0)).set_orientation(Vector.random_vector_xy())
        position_set_outofplane = lambda : particle.set_position(position).set_orientation(Vector.random_vector())
        self.particles[i] = (position_set_inplane if in_plane else position_set_outofplane)()
        
    def _force_particle_inside_box(self, i, half_test = False, in_plane = False) -> None:
        for _ in range(1000):
            if not self.wall_or_particle_collision(i, half_test=half_test):
                return
            self.move_inside_box(i, in_plane=in_plane)
        raise ValueError("Failed to find unexcluded particle configuration. Try lowering number of particles in box")
        
    def force_inside_box(self, in_plane: bool = False) -> None:
        for i, _ in enumerate(self.particles):
            self._force_particle_inside_box(i, half_test=False, in_plane=in_plane)
            

    def collision_test(self) -> bool:
        return any(self.wall_or_particle_collision(i, half_test=True) for i, _ in enumerate(self.particles))
     

    def get_nearest_particle(self, particle: Particle) -> Particle:
        def get_distance_from_particle(particle2: Particle):
            return np.inf if particle is particle2 else (particle.position - particle2.position).mag
        return min(self.particles, key = get_distance_from_particle)
        
    def plot_particle_positions(self, symbol: str = 'b.') -> None: # Mark for deletion
        plt.plot([p.position.x for p in self.particles], [p.position.y for p in self.particles], symbol)
        plt.show()

    def plot_particle_magnetizations(self) -> None: # Mark for deletion
        radius_from_particle = lambda p: (p.volume / (4 * PI/3))**(1/3)
        def arrow(particle: Particle) -> List[float]:
            magnetization_normalized = radius_from_particle(particle) * (particle.magnetization.unit_vector)
            return [particle.position.x, particle.position.y, magnetization_normalized.x, magnetization_normalized.y]
        for particle in self.particles:
            arrow_p = arrow(particle)
            plt.arrow(x = arrow_p[0], y = arrow_p[1], dx = arrow_p[2], dy=arrow_p[3])
        plt.show()

    
if __name__ == "__main__":
    pass

#%%