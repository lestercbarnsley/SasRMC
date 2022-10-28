from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

import numpy as np

from sas_rmc.shapes.shapes import Sphere

from .acceptance_scheme import AcceptanceScheme
from .scattering_simulation import MAGNETIC_RESCALE, NUCLEAR_RESCALE, ScatteringSimulation
from .box_simulation import Box
from .particles import Particle, CoreShellParticle
from .scattering_simulation import SimulationParams
from .vector import Vector

# execute should be a pure function, so I get rid of the instance of rng

def small_angle_change(vector: Vector, angle_change: float, reference_vector: Vector = None) -> Vector:
    ref_vec = reference_vector if reference_vector is not None else Vector.null_vector()
    delta_vector = vector - ref_vec
    angle = np.arctan2(delta_vector.y, delta_vector.x)
    mag = np.sqrt(delta_vector.x**2 + delta_vector.y**2 )
    new_angle = angle + angle_change
    new_vector = Vector(x = mag * np.cos(new_angle), y = mag * np.sin(new_angle), z = delta_vector.z)
    return new_vector + ref_vec


@dataclass
class Command(ABC):
    data: dict = field(default_factory = dict, init = False, repr = False)

    @abstractmethod
    def execute(self) -> None:
        pass

    @abstractmethod
    def _cls_specific_loggable_data(self) -> dict:
        return {}

    def update_loggable_data(self, data: dict) -> None:
        self.data.update(data)

    def get_loggable_data(self) -> dict:
        return {
            **self.data,
            **self._cls_specific_loggable_data(),
        }

    @abstractmethod
    def physical_acceptance_weak(self) -> bool:
        '''A weak physical acceptance test only tests the most recent command. It assumes all prior allowed commands are also physically acceptable'''
        pass



@dataclass
class ParticleCommand(Command):
    box: Box
    particle_index: int

    @property
    def particle(self) -> Particle:
        return self.box[self.particle_index]

    @abstractmethod
    def execute(self) -> None:
        return super().execute()

    def _cls_specific_loggable_data(self):
        particle = self.particle
        return {
            "Particle Index": self.particle_index,
            "Action": type(self).__name__,
            "Particle": type(particle).__name__,
            "Position.X": particle.position.x,
            "Position.Y": particle.position.y,
            "Position.Z": particle.position.z,
            "Orientation.X": particle.orientation.x,
            "Orientation.Y": particle.orientation.y,
            "Orientation.Z": particle.orientation.z,
            "Magnetization.X": particle._magnetization.x,
            "Magnetization.Y": particle._magnetization.y,
            "Magnetization.Z": particle._magnetization.z,
            "Volume": particle.volume,
            "Scattering Length": particle.scattering_length,
        }

    def physical_acceptance_weak(self) -> bool:
        return not self.box.wall_or_particle_collision(self.particle_index)



@dataclass
class SetParticleState(ParticleCommand):
    new_particle: Particle

    def execute(self) -> None:
        self.box.particles[self.particle_index] = self.new_particle

    @classmethod
    def gen_from_particle(cls, box: Box, particle_index: int):
        return cls(box, particle_index, new_particle = box[particle_index])



@dataclass
class MoveParticleTo(ParticleCommand):
    position_new: Vector

    def execute(self) -> None:
        particle = self.particle
        SetParticleState(self.box, self.particle_index, particle.set_position(self.position_new)).execute()
        
        

@dataclass
class MoveParticleBy(ParticleCommand):
    position_delta: Vector

    def execute(self) -> None:
        particle = self.particle
        position_new = particle.position + self.position_delta
        MoveParticleTo(self.box, self.particle_index, position_new).execute()


@dataclass
class JumpParticleTo(ParticleCommand):
    reference_particle_index: int

    def execute(self) -> None:
        MAX_MOVE = 1.00000005
        particle = self.particle
        reference_particle = self.box[self.reference_particle_index]
        jump_vector = Vector(np.inf, np.inf, np.inf)
        pointing_vector = reference_particle.closest_surface_position(particle.position)
        for shape in particle.shapes:
            for shape_2 in reference_particle.shapes:
                pointing_vector = shape_2.closest_surface_position(shape.central_position)
                reference_particle_dimension = (pointing_vector - shape_2.central_position).mag
                reverse_pointing_vector = shape.closest_surface_position(shape_2.central_position)
                particle_dimension = (reverse_pointing_vector - shape.central_position).mag
                gap_distance = MAX_MOVE * (reference_particle_dimension + particle_dimension)
                absolute_pointing_vector = shape_2.central_position - shape.central_position
                target_position = shape_2.central_position - gap_distance * (absolute_pointing_vector.unit_vector)
                jump_vector_new = target_position - shape.central_position
                jump_vector = jump_vector if jump_vector.mag < jump_vector_new.mag else jump_vector_new
        particle_move_command = MoveParticleBy(self.box, self.particle_index, position_delta=jump_vector)
        particle_move_command.execute()


@dataclass
class OrbitParticle(ParticleCommand):
    relative_angle: float

    def execute(self) -> None:
        particle = self.particle
        reference_particle = self.box.get_nearest_particle(particle)
        position_new = small_angle_change(particle.position, self.relative_angle, reference_particle.position)
        MoveParticleTo(self.box, self.particle_index, position_new).execute()


@dataclass
class ReorientateParticle(ParticleCommand):
    orientation_new: Vector

    def execute(self) -> None:
        particle = self.particle
        particle = self.particle
        SetParticleState(self.box, self.particle_index, particle.set_orientation(self.orientation_new)).execute()
        
        

@dataclass
class RotateParticle(ParticleCommand):
    relative_angle: float

    def execute(self) -> None:
        orientation_old = self.particle.orientation
        orientation_new = small_angle_change(orientation_old, self.relative_angle)
        ReorientateParticle(self.box, self.particle_index, orientation_new).execute()

@dataclass
class MagnetizeParticle(ParticleCommand):
    magnetization: Vector

    def execute(self) -> None:
        particle = self.particle
        SetParticleState(self.box, self.particle_index, particle.set_magnetization(self.magnetization)).execute()
        


@dataclass
class RotateMagnetization(ParticleCommand):
    relative_angle: float

    def execute(self) -> None:
        magnetization_old = self.particle.magnetization
        magnetization_new = small_angle_change(magnetization_old, self.relative_angle)
        MagnetizeParticle(self.box, self.particle_index, magnetization_new).execute()

@dataclass
class FlipMagnetization(ParticleCommand):
    def execute(self) -> None:
        magnetization_old = self.particle.magnetization
        magnetization_new = -1 * magnetization_old
        MagnetizeParticle(self.box, self.particle_index, magnetization_new).execute()

@dataclass
class CompressShell(ParticleCommand):
    change_by_factor: float
    reference_particle_index: int

    def execute(self) -> None:
        particle = self.particle
        if not isinstance(particle, CoreShellParticle):
            raise TypeError("This command can only be used with CoreShellParticle")
        core_radius = particle.core_sphere.radius
        new_thickness = self.change_by_factor * particle.shell_thickness
        old_scattering_length = particle.shell_sld * (particle.shell_sphere.volume - particle.core_sphere.volume)
        new_sld = old_scattering_length / (Sphere(radius = (core_radius + new_thickness)).volume - particle.core_sphere.volume)
        new_particle = CoreShellParticle.gen_from_parameters(
            position = particle.position,
            magnetization=particle.magnetization,
            core_radius=particle.core_sphere.radius,
            thickness=new_thickness,
            core_sld=particle.core_sld,
            shell_sld = new_sld,
            solvent_sld=particle.solvent_sld
        )
        SetParticleState(self.box, self.particle_index, new_particle).execute()
        JumpParticleTo(self.box, self.particle_index, self.reference_particle_index).execute()



@dataclass
class RescaleMagnetization(ParticleCommand):
    change_by_factor: float = 1

    def execute(self) -> None:
        magnetization_old = self.particle.magnetization
        magnetization_new = (self.change_by_factor * magnetization_old.mag) * magnetization_old.unit_vector
        MagnetizeParticle(self.box, self.particle_index, magnetization_new).execute()


@dataclass
class ScaleCommand(Command):
    simulation_params: SimulationParams

    @abstractmethod
    def execute(self) -> None:
        return super().execute()

    def _cls_specific_loggable_data(self) -> dict:
        return {
            "Particle Index": -1,
            "Action": type(self).__name__,
            **self.simulation_params.to_value_dict()

        }

    def physical_acceptance_weak(self) -> bool:
        return self.simulation_params.get_physical_acceptance()


@dataclass
class NuclearScale(ScaleCommand):
    change_to_factor: float

    def execute(self) -> None:
        self.simulation_params.set_value(key = NUCLEAR_RESCALE, value = self.change_to_factor)


@dataclass
class MagneticScale(ScaleCommand):
    change_to_factor: float

    def execute(self) -> None:
        self.simulation_params.set_value(key = MAGNETIC_RESCALE, value = self.change_to_factor)


@dataclass
class NuclearRescale(ScaleCommand):
    change_by_factor: float = 1

    def execute(self) -> None:
        nuclear_rescale = self.simulation_params.get_value(key = NUCLEAR_RESCALE)
        new_scale = nuclear_rescale * self.change_by_factor
        NuclearScale(self.simulation_params, change_to_factor=new_scale).execute()
        

@dataclass
class MagneticRescale(ScaleCommand):
    change_by_factor: float = 1

    def execute(self) -> None:
        magnetic_rescale = self.simulation_params.get_value(key = MAGNETIC_RESCALE)
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
        rescale_value = self.simulation_params.get_value(key = NUCLEAR_RESCALE)
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
        #command_ledger.append(SetSimulationParams(simulation_params, change_to_factors=simulation_params.values))
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
