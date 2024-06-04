from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import ParamSpec, TypeVar

import numpy as np

from sas_rmc.particles import Particle
from sas_rmc.scattering_simulation import ScatteringSimulation, SimulationParam
from sas_rmc import Vector

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
    document: dict | None = field(default_factory=lambda : None, repr=False, init=False)

    @abstractmethod
    def execute(self, scattering_simulation: ScatteringSimulation | None = None) -> ScatteringSimulation:
        pass

    def get_document(self) -> dict:
        if self.document is None:
            return {}
        return self.document



@dataclass
class ParticleCommand(Command):
    box_index: int
    particle_index: int

    def get_particle(self, scattering_simulation: ScatteringSimulation) -> Particle:
        return scattering_simulation.get_particle(self.box_index, self.particle_index)
    
    def document_from_command(self, command: ParticleCommand, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        new_scattering_simulation = command.execute(scattering_simulation)
        self.document = command.get_document()
        return new_scattering_simulation

    @abstractmethod
    def execute(self, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        if scattering_simulation is None:
            raise ValueError("scattering_simulation is not optional for this class")
        particle = self.get_particle(scattering_simulation)
        self.document = {
            "Action" : type(self).__name__,
            "Box index" : self.box_index,
            "Particle index" : self.particle_index } | particle.get_loggable_data()
        return scattering_simulation



R = TypeVar("R")
P = ParamSpec("P")

def generate_document(func: Callable[P, R]) -> Callable[P, R]:

    @wraps(func)
    def execute_with_document(*args: P.args, **kwargs: P.kwargs) -> R:
        result = func(*args, **kwargs)
        obj = args[0]
        if not isinstance(obj, ParticleCommand):
            raise TypeError("inappropriate use of generate document decorator")
        particle = obj.get_particle(result)
        obj.document = {
            "Action" : type(obj).__name__,
            "Box index" : obj.box_index,
            "Particle index" : obj.particle_index } | particle.get_loggable_data()
        return result
    return execute_with_document


@dataclass
class SetParticleState(ParticleCommand):
    new_particle: Particle

    def execute(self, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        new_scattering_simulation = scattering_simulation.change_particle(self.box_index, self.particle_index, self.new_particle)
        new_scattering_simulation = super().execute(new_scattering_simulation)
        return new_scattering_simulation


@dataclass
class MoveParticleTo(ParticleCommand):
    position_new: Vector

    def execute(self, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        particle = self.get_particle(scattering_simulation)
        new_particle = particle.change_position(self.position_new)
        command = SetParticleState(self.box_index, self.particle_index, new_particle)
        return self.document_from_command(command, scattering_simulation)


@dataclass
class MoveParticleBy(ParticleCommand):
    position_delta: Vector

    def execute(self, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        particle = self.get_particle(scattering_simulation)
        position_new = particle.get_position() + self.position_delta
        command = MoveParticleTo(self.box_index, self.particle_index, position_new)
        return self.document_from_command(command, scattering_simulation)

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
        return self.document_from_command(particle_move_command, scattering_simulation)


def jump_by_vector(position_1: Vector, position_2: Vector, fixed_distance: float) -> Vector:
    pointing_vector = position_2 - position_1
    return pointing_vector - (fixed_distance * pointing_vector.unit_vector)




def rotate_vector(vector: Vector, angle: float) -> Vector:
    x, y, z = vector.to_tuple()
    if z:
        raise ValueError("remember, no z")
    return Vector(
        x = x * np.cos(angle) - y * np.sin(angle),
        y = x * np.sin(angle) + y * np.cos(angle)
    )


@dataclass
class FormLattice(ParticleCommand):
    reference_particle_index: int
    reference_angle: float

    def execute(self) -> None:
        reference_particle = self.box[self.reference_particle_index]
        pointing_vectors = ((particle.position - reference_particle.position) for particle in self.box.particles)
        vector_to_reference = min(pointing_vectors, key= lambda v : v.mag)
        new_position = reference_particle.position + rotate_vector(vector_to_reference, self.reference_angle)
        MoveParticleTo(self.box, self.particle_index, new_position).execute()


@dataclass
class OrbitParticle(ParticleCommand):
    relative_angle: float

    @generate_document
    def execute(self, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        particle = self.get_particle(scattering_simulation)
        reference_particle = scattering_simulation.box_list[self.box_index].get_nearest_particle(particle.get_position())
        position_new = small_angle_change(particle.get_position(), self.relative_angle, reference_particle.get_position())
        return MoveParticleTo(self.box_index, self.box_index, position_new).execute(scattering_simulation)


@dataclass
class ReorientateParticle(ParticleCommand):
    orientation_new: Vector

    @generate_document
    def execute(self, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        particle = self.get_particle(scattering_simulation)
        new_particle = particle.change_orientation(self.orientation_new)
        return SetParticleState(self.box_index, self.particle_index, new_particle).execute(scattering_simulation)        
        

@dataclass
class RotateParticle(ParticleCommand):
    relative_angle: float
    
    @generate_document
    def execute(self, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        particle = self.get_particle(scattering_simulation)
        orientation_old = particle.get_orientation()
        orientation_new = small_angle_change(orientation_old, self.relative_angle)
        new_particle = particle.change_orientation(orientation_new)
        return ReorientateParticle(self.box_index, self.particle_index, new_particle).execute(scattering_simulation)


@dataclass
class MagnetizeParticle(ParticleCommand):
    magnetization: Vector

    @generate_document
    def execute(self, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        particle = self.get_particle(scattering_simulation)
        new_particle = particle.change_magnetization(self.magnetization)
        return SetParticleState(self.box_index, self.particle_index, new_particle).execute(scattering_simulation)
        

@dataclass
class RotateMagnetization(ParticleCommand):
    relative_angle: float

    @generate_document
    def execute(self, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        particle = self.get_particle(scattering_simulation)
        magnetization = particle.get_magnetization()
        magnetization_new = small_angle_change(magnetization, self.relative_angle)
        return MagnetizeParticle(self.box_index, self.particle_index, magnetization_new).execute(scattering_simulation)

@dataclass
class FlipMagnetization(ParticleCommand):

    @generate_document
    def execute(self, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        particle = self.get_particle(scattering_simulation)
        magnetization = particle.get_magnetization()
        magnetization_new = -1 * magnetization
        return MagnetizeParticle(self.box_index, self.particle_index, magnetization_new).execute(scattering_simulation)

@dataclass
class ScaleCommand(Command):
    scale_factor: float

    def execute(self, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        self.set_document()
        return scattering_simulation.set_scale_factor(self.scale_factor)
    
    def set_document(self):
        self.document = {
            "Action" : type(self).__name__,
            "Box index" : -1,
            "Particle index" : -1,
            "Scale factor" : self.scale_factor,
            }

@dataclass
class RelativeRescale(ScaleCommand):
    change_by_factor: float = 1

    def execute(self, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        current_scale_factor = scattering_simulation.scale_factor.value
        new_scale_factor = self.change_by_factor * current_scale_factor
        scale_command = ScaleCommand(new_scale_factor)
        res = scale_command.execute(scattering_simulation)
        self.document = scale_command.document
        return res


@dataclass
class MagneticRescale(ScaleCommand):
    change_by_factor: float = 1

    def execute(self) -> None:
        magnetic_rescale = self.simulation_params.get_value(key = constants.MAGNETIC_RESCALE)
        new_scale = magnetic_rescale * self.change_by_factor
        MagneticScale(self.simulation_params, change_to_factor=new_scale).execute()


@dataclass
class NuclearMagneticScale(ScaleCommand):
    change_to_factor: float

    def execute(self) -> None:
        NuclearScale(self.simulation_params, change_to_factor=self.change_to_factor).execute()
        MagneticScale(self.simulation_params, change_to_factor=self.change_to_factor).execute()


@dataclass
class NuclearMagneticRescale(ScaleCommand):
    change_by_factor: float = 1

    def execute(self) -> None:
        rescale_value = self.simulation_params.get_value(key = constants.NUCLEAR_RESCALE)
        new_scale_factor = rescale_value * self.change_by_factor
        NuclearMagneticScale(self.simulation_params, change_to_factor=new_scale_factor).execute()


@dataclass
class SetSimulationParams(ScaleCommand):
    change_to_factors: List[float]

    def execute(self) -> None:
        for change_to_factor, simulation_param in zip(self.change_to_factors, self.simulation_params.params):
            simulation_param.set_value(change_to_factor)

    @classmethod
    def gen_from_simulation_params(cls, simulation_params: SimulationParams):
        return cls(simulation_params=simulation_params, change_to_factors=simulation_params.values)


@dataclass
class SetSimulationState(Command):
    simulation_params: SimulationParams
    _command_ledger: List[Command]

    def execute(self) -> None:
        for command in self._command_ledger:
            command.execute()

    def physical_acceptance_weak(self) -> bool:
        return self.simulation_params.get_physical_acceptance()

    def _cls_specific_loggable_data(self) -> dict:
        return super()._cls_specific_loggable_data() # I want to implement this, but it has to return a "square" dict

    @classmethod
    def gen_from_simulation(cls, simulation_params: SimulationParams, box_list: List[Box]):
        command_ledger = [SetSimulationParams.gen_from_simulation_params(simulation_params)]
        for box in box_list: # Google Python style guide says to write it this way rather than a 2-iterator list comprehension
            for particle_index, _ in enumerate(box.particles):
                command_ledger.append(
                    SetParticleState.gen_from_particle(
                        box = box,
                        particle_index=particle_index
                    )
                )
        return cls(simulation_params = simulation_params, _command_ledger = command_ledger)



@dataclass
class AcceptableCommand:
    base_command: Command
    acceptance_scheme: AcceptanceScheme
    
    def execute(self) -> None:
        acceptable = True if self.acceptance_scheme is None else self.acceptance_scheme.is_acceptable() # This is the only thing that justifies this class' existance
        if acceptable:
            self.base_command.execute()

    def _cls_specific_loggable_data(self) -> dict:
        return {
            **self.base_command.get_loggable_data(),
            **self.acceptance_scheme.get_loggable_data(),
        }

    def get_loggable_data(self) -> dict:
        return self._cls_specific_loggable_data()

    def update_loggable_data(self, data: dict) -> None:
        self.base_command.update_loggable_data(data)

    def physical_acceptance_weak(self) -> bool:
        return self.base_command.physical_acceptance_weak()

    def handle_simulation(self, simulation: ScatteringSimulation) -> None:
        self.acceptance_scheme.set_physical_acceptance(self.physical_acceptance_weak())
        self.acceptance_scheme.handle_simulation(simulation=simulation)
