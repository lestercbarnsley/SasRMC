#%%

from dataclasses import dataclass

from typing_extensions import Self
import numpy as np

from sas_rmc import Particle, Vector
from sas_rmc.particles import CoreShellParticle
from sas_rmc.shapes import Shape


def numerical_form(x_array: np.ndarray, y_array: np.ndarray, sld_sum: np.ndarray, qx: float, qy: float) -> float:
    return np.sum(sld_sum * np.exp(1j * (x_array * qx + y_array * qy)))

def numerical_form_array(x_array: np.ndarray, y_array: np.ndarray, sld_sum: np.ndarray, qx_array: np.ndarray, qy_array: np.ndarray) -> np.ndarray
    form = np.zeros(qx_array.shape)
    for ji, qx in np.ndenumerate(qx_array):
        j, i = ji
        qy = qy_array[j, i]
        form[j, i] = numerical_form(x_array, y_array, sld_sum, qx, qy)
    return form


@dataclass
class Dumbbell(Particle):
    core_particle: CoreShellParticle
    seed_particle: CoreShellParticle
    simulation_lattice_spacing: float

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
            simulation_lattice_spacing=self.simulation_lattice_spacing
        )
    
    def change_orientation(self, orientation: Vector) -> Self:
        distance_between_particles = (self.core_particle.get_position() - self.seed_particle.get_position()).mag
        seed_particle_position = self.get_position() + distance_between_particles * (orientation.unit_vector)
        return type(self)(
            core_particle=self.core_particle,
            seed_particle=self.seed_particle.change_position(seed_particle_position),
            simulation_lattice_spacing=self.simulation_lattice_spacing
        )
    
    def change_core_magnetization(self, magnetization: Vector) -> Self:
        return type(self)(
            core_particle=self.core_particle.change_magnetization(magnetization),
            seed_particle=self.seed_particle,
            simulation_lattice_spacing=self.simulation_lattice_spacing
        )
    
    def change_seed_magnetization(self, magnetization: Vector) -> Self:
        return type(self)(
            core_particle=self.core_particle,
            seed_particle=self.seed_particle.change_magnetization(magnetization),
            simulation_lattice_spacing=self.simulation_lattice_spacing
        )
    
    def change_magnetization(self, magnetization: Vector) -> Self:
        return self.change_core_magnetization(magnetization)
    
    def get_loggable_data(self) -> dict:
        return super().get_loggable_data() | {
            'Core radius': self.core_particle.core_radius,
            'Seed radius' : self.seed_particle.core_radius,
            'Shell thickness': self.core_particle.thickness,
            'Core SLD': self.core_particle.core_sld,
            'Seed SLD': self.seed_particle.core_sld,
            'Shell SLD' : self.core_particle.shell_sld,
            'Solvent SLD': self.core_particle.solvent_sld,
        }
    
    def get_sld(self, position: Vector) -> float:
        position_delta_offset = position + self.get_position()
        solvent_sld = self.core_particle.solvent_sld
        if self.core_particle.core_sphere.is_inside(position_delta_offset):
            return self.core_particle.core_sld - solvent_sld
        if self.seed_particle.core_sphere.is_inside(position_delta_offset):
            return self.seed_particle.core_sld - solvent_sld
        if self.is_inside(position_delta_offset):
            return self.core_particle.shell_sld - solvent_sld
        return solvent_sld
        
    
    def form_array(self, qx_array: np.ndarray, qy_array: np.ndarray) -> np.ndarray:
        particle_size = 2 * self.core_particle.core_radius
        x_array, y_array = np.meshgrid(
            np.arange(start = -2 * particle_size, stop= 2* particle_size, step = self.simulation_lattice_spacing),
            np.arange(start = -2 * particle_size, stop= 2* particle_size, step = self.simulation_lattice_spacing))
        z_array = np.arange(start = -2 * particle_size, stop = 2* particle_size, step=self.simulation_lattice_spacing)
        sld = np.zeros(x_array.shape)
        for ji, x in np.ndenumerate(x_array):
            j, i = ji
            y = y_array[j, i]
            sld[j,i] = np.sum([self.get_sld(Vector(x, y, z)) * (self.simulation_lattice_spacing)**3 for z in z_array])
        return numerical_form_array(x_array, y_array, sld, qx_array, qy_array)
        
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
        return cls(
            core_particle=core_shell,
            seed_particle=seed_shell,
            simulation_lattice_spacing=100
        ).change_orientation(orientation)


if __name__ == "__main__": 
    pass

