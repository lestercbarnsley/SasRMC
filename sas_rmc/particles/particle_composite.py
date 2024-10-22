#%%
from abc import abstractmethod
from dataclasses import dataclass, field

import numpy as np

from sas_rmc import Shape, Particle, Vector



@dataclass
class ParticleComposite(Particle):
    particle_list: list[Particle] = field(default_factory=list)

    def get_position(self) -> Vector:
        position_vector = Vector.null_vector()
        for particle in self.particle_list:
            position_vector = position_vector + particle.get_position()
        return position_vector / len(self.particle_list)

    @property
    def shapes(self) -> List[Shape]:
        shapes_list = []
        for particle in self.particle_list:
            for shape in particle.shapes:
                shapes_list.append(shape)
        return shapes_list

    def _change_shapes(self, shape_list: List[Shape]):
        return super()._change_shapes(shape_list)

    @abstractmethod
    def change_particle_list(self, particle_list: List[Particle]):
        pass

    @property
    def scattering_length(self) -> float:
        return np.sum([particle.scattering_length for particle in self.particle_list])

    @property
    def volume(self) -> float:
        return np.sum([particle.volume for particle in self.particle_list])

    @property
    def position(self) -> Vector:
        return self.particle_list[0]

    def set_position(self, position: Vector):
        position_change = position - self.position
        particle_list = [particle.set_position(particle.position + position_change) for particle in self.particle_list]
        return self.change_particle_list(particle_list)

    @property
    def orientation(self) -> Vector:
        return (self.particle_list[-1].position - self.particle_list[0].position).unit_vector

    @abstractmethod
    def set_orientation(self, orientation: Vector) -> None:
        pass

    def is_spherical(self) -> bool:
        return all(round_vector(particle.position) == round_vector(self.position) for particle in self.particle_list)

    def is_inside(self, position: Vector) -> bool:
        return any(particle.is_inside(position) for particle in self.particle_list)

    def form_array(self, qx_array: np.ndarray, qy_array: np.ndarray, orientation: Vector) -> np.ndarray:
        return super().form_array(qx_array, qy_array, orientation) # Theoretically, this doesn't get called

    def magnetic_form_array(self, qx_array: np.ndarray, qy_array: np.ndarray, orientation: Vector, magnetization: Vector) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return super().magnetic_form_array(qx_array, qy_array, orientation, magnetization) # Theoretically, this doesn't get called

    def random_position_inside(self) -> Vector:
        random_particle = np.random.choice(self.particle_list)
        return random_particle.random_position_inside()

    def collision_detected(self, other_particle: Particle) -> bool:
        if isinstance(other_particle, ParticleComposite):
            return any(any(particle.collision_detected(other_particle_comp) for other_particle_comp in other_particle.particle_list) for particle in self.particle_list)
        return any(particle.collision_detected(other_particle) for particle in self.particle_list)

    def closest_surface_position(self, position: Vector) -> Vector:
        surface_positions = [particle.closest_surface_position(position) for particle in self.particle_list]
        return min(surface_positions, key = lambda surface_position : (surface_position - position).mag)
        #return surface_positions[np.argmin([(surface_position - position).mag for surface_position in surface_positions])]


if __name__ == "__main__":
    pass


