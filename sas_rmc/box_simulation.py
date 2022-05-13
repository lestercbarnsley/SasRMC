#%%

from dataclasses import dataclass, field
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from .particle import Particle
from .vector import Vector
from .shapes import Cube


PI = np.pi
rng = np.random.default_rng()


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
        return any([particle.is_magnetic() for particle in self.particles])
    
    def wall_or_particle_collision(self, i: int, half_test = False) -> bool:
        compare_to_i = lambda j: (i < j) if half_test else (i != j)
        particle = self.particles[i]
        if not self.is_inside(particle.position):
            return True
        for j, particle2 in enumerate(self.particles):
            if compare_to_i(j) and particle.collision_detected(particle2):
                return True
        return False

    def move_inside_box(self, i: int, in_plane: bool = False) -> None:
        particle = self.particles[i]
        position = self.cube.random_position_inside()
        if in_plane:
            particle.set_position(Vector(position.x, position.y, z = 0))
            particle.set_orientation(Vector.random_vector_xy())
        else:
            particle.set_position(position)
            particle.set_orientation(Vector.random_vector())

    def _force_particle_inside_box(self, i, half_test = False, in_plane = False) -> None:
        for _ in range(100000):
            if not self.wall_or_particle_collision(i, half_test=half_test):
                return
            self.move_inside_box(i, in_plane=in_plane)
        raise Exception("Failed to find unexcluded particle configuration. Try lowering number of particles in box")
        
    def force_inside_box(self, in_plane: bool = False) -> None:
        for i, _ in enumerate(self.particles):
            self._force_particle_inside_box(i, half_test=True, in_plane=in_plane)
            
    def collision_test(self) -> bool:
        for i, _ in enumerate(self.particles):
            if self.wall_or_particle_collision(i, half_test=True):
                return True
        return False

    def get_nearest_particle(self, particle: Particle) -> Particle:
        #particle = self.particles[particle_index]
        def distance(particle2: Particle):
            return np.inf if particle is particle2 else (particle.position - particle2.position).mag
        distances = [distance(particle2) for particle2 in self.particles]
        return self.particles[np.argmin(distances)]

    def plot_particle_positions(self, symbol: str = 'b.') -> None:
        plt.plot([p.position.x for p in self.particles], [p.position.y for p in self.particles], symbol)
        plt.show()

    def plot_particle_magnetizations(self) -> None:
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