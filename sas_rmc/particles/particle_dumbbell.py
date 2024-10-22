#%%

from dataclasses import dataclass

from typing_extensions import Self

from sas_rmc import Particle, Vector
from sas_rmc.particles import CoreShellParticle
from sas_rmc.shapes import Shape


@dataclass
class Dumbbell(Particle):
    core_particle: CoreShellParticle
    seed_particle: CoreShellParticle
    simulation_density: float

    def get_position(self) -> Vector:
        return self.core_particle.get_position()
    
    def get_orientation(self) -> Vector:
        return (self.seed_particle.get_position() - self.core_particle.get_position()).unit_vector
    
    def get_magnetization(self) -> Vector:
        return max([self.core_particle.get_magnetization(), self.seed_particle.get_magnetization()], key = lambda v : v.mag)
    
    def get_volume(self) -> float:
        return self.core_particle.get_volume() + self.seed_particle.get_volume()
    
    def get_shapes(self) -> list[Shape]:
        return self.core_particle.get_shapes() + self.seed_particle.get_shapes()
    
    def is_inside(self, position: Vector) -> bool:
        return self.core_particle.is_inside(position) or self.seed_particle.is_inside(position)
    
    def collision_detected(self, other_particle: Self) -> bool:
        return self.core_particle.collision_detected(other_particle) or self.seed_particle.collision_detected(other_particle)
    
    def get_scattering_length(self) -> float:
        return self.core_particle.get_scattering_length() + self.seed_particle.get_scattering_length()
    
    def is_magnetic(self) -> bool:
        return self.core_particle.is_magnetic() or self.seed_particle.is_magnetic()
    
    def change_position(self, position: Vector) -> Self:
        position_delta = position - self.get_position()
        return type(self)(
            core_particle=self.core_particle.change_position(self.core_particle.get_position() + position_delta),
            seed_particle=self.seed_particle.change_position(self.seed_particle.get_position() + position_delta),
            simulation_density=self.simulation_density
        )
    
    def change_orientation(self, orientation: Vector) -> Self:
        distance_between_particles = (self.core_particle.get_position() - self.seed_particle.get_position()).mag
        seed_particle_position = self.get_position() + distance_between_particles * (orientation.unit_vector)
        return type(self)(
            core_particle=self.core_particle,
            seed_particle=self.seed_particle.change_position(seed_particle_position),
            simulation_density=self.simulation_density
        )
    
    def change_magnetization(self, magnetization: Vector) -> Self:
        if magnetization.mag == 0:
            return type(self)(
                core_particle=self.core_particle.change_magnetization(magnetization),
                seed_particle=self.seed_particle.change_magnetization(magnetization),
                simulation_density=self.simulation_density
            )
        magnetization_factor = self.core_particle.get_magnetization().mag / magnetization.mag


    def change_particle_list(self, particle_list: list[Particle]):
        return Dumbbell(
            _magnetization = self._magnetization,
            _shapes = self._shapes,
            solvent_sld=self.solvent_sld,
            particle_list=particle_list,
            _centre_to_centre_distance = self._centre_to_centre_distance
        )
    
    def is_spherical(self) -> bool:
        return False

    

    @property
    def centre_to_centre_distance(self) -> float:
        return self._centre_to_centre_distance

    @property
    def position(self) -> Vector:
        return self.core_particle.position

    @property
    def orientation(self) -> Vector:
        return (self.seed_particle.position - self.core_particle.position).unit_vector

    def set_orientation(self, orientation: Vector):
        orientation_new = orientation.unit_vector
        distance = self.centre_to_centre_distance
        new_seed = self.seed_particle.set_position(self.core_particle.position + distance * orientation_new)
        particle_list = [self.core_particle, new_seed]
        return self.change_particle_list(particle_list)

    def set_position(self, position: Vector):
        return super().set_position(position)

    def set_centre_to_centre_distance(self, centre_to_centre_distance: float):
        orientation_new = self.orientation.unit_vector
        distance = centre_to_centre_distance
        new_seed = self.seed_particle.set_position(self.core_particle.position + distance * orientation_new)
        particle_list = [self.core_particle, new_seed]
        return self.change_particle_list(particle_list)

    def get_sld(self, relative_position: Vector) -> float:
        position = relative_position + self.position
        if self.seed_particle.core_sphere.is_inside(position): # The seed particle is the spherical one
            return self.seed_particle.core_sld
        if self.core_particle.core_sphere.is_inside(position):
            return self.core_particle.core_sld
        if self.is_inside(position):
            return self.core_particle.shell_sld
        return self.solvent_sld

    ''' def get_magnetization(self, relative_position: Vector) -> Vector:
        position = relative_position + self.position
        if self.seed_particle.core_sphere.is_inside(position):
            return self.seed_particle.magnetization
        if self.core_particle.core_sphere.is_inside(position):
            return self.core_particle.magnetization
        return Vector.null_vector()'''

    @property
    def magnetization(self) -> Vector:
        return self.core_particle.magnetization

    def set_magnetization(self, magnetization: Vector):
        core_particle = self.core_particle.set_magnetization(magnetization)
        return self.change_particle_list([core_particle, self.seed_particle])

    @property
    def seed_magnetization(self) -> Vector:
        return self.seed_particle.magnetization

    def set_seed_magnetization(self, magnetization: Vector):
        seed_particle = self.seed_particle.set_magnetization(magnetization)
        return self.change_particle_list([self.core_particle, seed_particle])

    def get_loggable_data(self) -> dict:
        core_radius = self.core_particle.shapes[0].radius
        overall_radius = self.core_particle.shapes[1].radius
        thickness = overall_radius - core_radius
        data = {
            'Particle type' : "",
            'Core radius': core_radius,
            'Seed radius' : self.seed_particle.shapes[0].radius,
            'Shell thickness': thickness,
            'Core SLD': self.core_particle.core_sld,
            'Seed SLD': self.seed_particle.core_sld,
            'Shell SLD' : self.core_particle.shell_sld,
            'Solvent SLD': self.solvent_sld,
        }
        data.update(super().get_loggable_data())
        data.update(self.core_particle.magnetization.to_dict('MagnetizationCore'))
        data.update(self.seed_particle.magnetization.to_dict('MagnetizationSeed'))
        
        return data

    @classmethod
    def gen_from_parameters(cls, core_radius, seed_radius, shell_thickness, core_sld, seed_sld, shell_sld, solvent_sld, position: Vector = None, centre_to_centre_distance: float = None, orientation: Vector = None, core_magnetization: Vector = None, seed_magnetization: Vector = None):
        if position is None:
            position = Vector.null_vector()
        if orientation is None:
            orientation = Vector(0, 1, 0)
        if centre_to_centre_distance is None:
            centre_to_centre_distance = core_radius + seed_radius + 2 * shell_thickness
        core_shell = CoreShellParticle.gen_from_parameters(
            position=position,
            core_radius=core_radius,
            thickness=shell_thickness,
            core_sld=core_sld,
            shell_sld=shell_sld,
            solvent_sld=solvent_sld,
            magnetization=core_magnetization if core_magnetization else Vector.null_vector()
        )
        seed_shell = CoreShellParticle.gen_from_parameters(
            position=position,
            core_radius=seed_radius,
            thickness=shell_thickness,
            core_sld=seed_sld,
            shell_sld=shell_sld,
            solvent_sld=solvent_sld,
            magnetization=seed_magnetization if seed_magnetization else Vector.null_vector()
        )
        dumbell = Dumbbell(
            solvent_sld=solvent_sld,
            particle_list=[core_shell, seed_shell],
            _centre_to_centre_distance = centre_to_centre_distance
        )
        return dumbell.set_orientation(orientation)
        #return dumbell


if __name__ == "__main__":
    pass

