from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from sas_rmc.scattering_simulation import ScatteringSimulation
from sas_rmc import Vector, Particle

# execute should be a pure function, so I get rid of the instance of rng

def small_angle_change(vector: Vector, angle_change: float, reference_vector: Vector | None = None) -> Vector:
    reference_vector = reference_vector if reference_vector is not None else Vector.null_vector()
    delta_vector = vector - reference_vector
    angle = np.arctan2(delta_vector.y, delta_vector.x)
    mag = Vector(x = delta_vector.x, y = delta_vector.y, z = 0).mag
    new_angle = angle + angle_change
    new_vector = Vector(x = mag * np.cos(new_angle), y = mag * np.sin(new_angle), z = delta_vector.z)
    return new_vector + reference_vector


@dataclass
class Command(ABC):
    
    @abstractmethod
    def execute(self, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        pass

    @abstractmethod
    def execute_and_get_document(self, scattering_simulation: ScatteringSimulation) -> tuple[ScatteringSimulation, dict]:
        pass


@dataclass
class ParticleCommand(Command):
    box_index: int
    particle_index: int

    def get_particle(self, scattering_simulation: ScatteringSimulation) -> Particle:
        return scattering_simulation.get_particle(self.box_index, self.particle_index)

    @abstractmethod
    def execute(self, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        pass
    
    def get_document_from_scattering_simulation(self, scattering_simulation: ScatteringSimulation) -> dict:
        particle = self.get_particle(scattering_simulation)
        return {
            "Action" : type(self).__name__,
            "Box index" : self.box_index,
            "Particle index" : self.particle_index } \
            | particle.get_loggable_data()
    
    def execute_and_get_document(self, scattering_simulation: ScatteringSimulation) -> tuple[ScatteringSimulation, dict]:
        new_simulation = self.execute(scattering_simulation)
        document = self.get_document_from_scattering_simulation(new_simulation)
        return new_simulation, document
    

@dataclass
class GroupCommand(Command):
    command_list: list[Command]

    def execute(self, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        return super().execute(scattering_simulation)

    def execute_and_get_document(self, scattering_simulation: ScatteringSimulation) -> tuple[ScatteringSimulation, dict]:
        document = {"Action" : type(self).__name__}
        new_simulation = scattering_simulation
        for i, command in enumerate(self.command_list):
            new_simulation, document_i = command.execute_and_get_document(new_simulation)
            document = document | {f"Command {i}" : document_i}
        return new_simulation, document


@dataclass
class SetParticleState(ParticleCommand):
    new_particle: Particle

    def execute(self, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        return scattering_simulation.change_particle(self.box_index, self.particle_index, self.new_particle) 


@dataclass
class MoveParticleTo(ParticleCommand):
    position_new: Vector

    def execute(self, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        particle = self.get_particle(scattering_simulation)
        new_particle = particle.change_position(self.position_new)
        return SetParticleState(self.box_index, self.particle_index, new_particle).execute(scattering_simulation)


@dataclass
class MoveParticleBy(ParticleCommand):
    position_delta: Vector

    def execute(self, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        particle = self.get_particle(scattering_simulation)
        position_new = particle.get_position() + self.position_delta
        return MoveParticleTo(self.box_index, self.particle_index, position_new).execute(scattering_simulation)


@dataclass
class JumpParticleTo(ParticleCommand):
    reference_particle_index: int

    def execute(self, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        MAX_MOVE = 1.00000005
        particle = self.get_particle(scattering_simulation)
        reference_particle = scattering_simulation.get_particle(self.box_index, self.reference_particle_index)
        jump_vector = Vector(np.inf, np.inf, np.inf)
        pointing_vector = reference_particle.get_position() - particle.get_position()
        for shape in particle.get_shapes():
            for shape_2 in reference_particle.get_shapes():
                pointing_vector = shape_2.closest_surface_position(shape.get_position())
                reference_particle_dimension = (pointing_vector - shape_2.get_position()).mag
                reverse_pointing_vector = shape.closest_surface_position(shape_2.get_position())
                particle_dimension = (reverse_pointing_vector - shape.get_position()).mag
                gap_distance = MAX_MOVE * (reference_particle_dimension + particle_dimension)
                absolute_pointing_vector = shape_2.get_position() - shape.get_position()
                target_position = shape_2.get_position() - gap_distance * (absolute_pointing_vector.unit_vector)
                jump_vector_new = target_position - shape.get_position()
                jump_vector = jump_vector if jump_vector.mag < jump_vector_new.mag else jump_vector_new
        particle_move_command = MoveParticleBy(self.box_index, self.particle_index, position_delta=jump_vector)
        return particle_move_command.execute(scattering_simulation)


@dataclass
class OrbitParticle(ParticleCommand):
    relative_angle: float

    def execute(self, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        particle = self.get_particle(scattering_simulation)
        reference_particle = scattering_simulation.box_list[self.box_index].get_nearest_particle(particle.get_position())
        position_new = small_angle_change(particle.get_position(), self.relative_angle, reference_particle.get_position())
        return MoveParticleTo(self.box_index, self.box_index, position_new).execute(scattering_simulation)


@dataclass
class ReorientateParticle(ParticleCommand):
    orientation_new: Vector

    def execute(self, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        particle = self.get_particle(scattering_simulation)
        new_particle = particle.change_orientation(self.orientation_new)
        return SetParticleState(self.box_index, self.particle_index, new_particle).execute(scattering_simulation)        
        

@dataclass
class RotateParticle(ParticleCommand):
    relative_angle: float
    
    def execute(self, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        particle = self.get_particle(scattering_simulation)
        orientation_old = particle.get_orientation()
        orientation_new = small_angle_change(orientation_old, self.relative_angle)
        return ReorientateParticle(self.box_index, self.particle_index, orientation_new).execute(scattering_simulation)


@dataclass
class MagnetizeParticle(ParticleCommand):
    magnetization: Vector

    def execute(self, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        particle = self.get_particle(scattering_simulation)
        new_particle = particle.change_magnetization(self.magnetization)
        return SetParticleState(self.box_index, self.particle_index, new_particle).execute(scattering_simulation)
        

@dataclass
class RescaleMagnetization(ParticleCommand):
    rescale_factor: float

    def execute(self, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        particle = self.get_particle(scattering_simulation)
        magnetization = particle.get_magnetization()
        new_magnetization = self.rescale_factor * magnetization
        return MagnetizeParticle(self.box_index, self.particle_index, new_magnetization).execute(scattering_simulation)


@dataclass
class RotateMagnetization(ParticleCommand):
    relative_angle: float

    def execute(self, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        particle = self.get_particle(scattering_simulation)
        magnetization = particle.get_magnetization()
        magnetization_new = small_angle_change(magnetization, self.relative_angle)
        return MagnetizeParticle(self.box_index, self.particle_index, magnetization_new).execute(scattering_simulation)


@dataclass
class FlipMagnetization(ParticleCommand):

    def execute(self, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        particle = self.get_particle(scattering_simulation)
        magnetization = particle.get_magnetization()
        magnetization_new = -1 * magnetization
        return MagnetizeParticle(self.box_index, self.particle_index, magnetization_new).execute(scattering_simulation)
    

@dataclass
class MutateParticle(ParticleCommand):
    particle_mutation_function: Callable[[Particle], Particle]

    def execute(self, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        particle = self.get_particle(scattering_simulation)
        new_particle = self.particle_mutation_function(particle)
        return SetParticleState(self.box_index, self.particle_index, new_particle).execute(scattering_simulation)


@dataclass
class ScaleCommand(Command):

    @abstractmethod
    def execute(self, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        pass
    
    def get_document_from_scattering_simulation(self, scattering_simulation: ScatteringSimulation) -> dict:
        scale_factor = scattering_simulation.scale_factor
        return {
            "Action" : type(self).__name__,
            "Box index" : -1,
            "Particle index" : -1} \
            | scale_factor.get_loggable_data()
    
    def execute_and_get_document(self, scattering_simulation: ScatteringSimulation) -> tuple[ScatteringSimulation, dict]:
        new_simulation = self.execute(scattering_simulation)
        document = self.get_document_from_scattering_simulation(new_simulation)
        return new_simulation, document


@dataclass
class RescaleCommand(ScaleCommand):
    scale_factor: float

    def execute(self, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        return scattering_simulation.set_scale_factor(self.scale_factor)
    

@dataclass
class RelativeRescale(ScaleCommand):
    change_by_factor: float = 1

    def execute(self, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        scale_factor = scattering_simulation.scale_factor.value
        return RescaleCommand(scale_factor=self.change_by_factor*scale_factor).execute(scattering_simulation)


#%%

if __name__ == "__main__":
    pass