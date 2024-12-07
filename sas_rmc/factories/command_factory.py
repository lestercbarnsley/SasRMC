#%%
import random

import pandas as pd
from pydantic.dataclasses import dataclass as pydantic_dataclass

from sas_rmc import commands, Vector, constants
from sas_rmc.factories import parse_data
from sas_rmc.particles import Particle
from sas_rmc.particles.particle_core_shell_spherical import CoreShellParticle
from sas_rmc.scattering_simulation import ScatteringSimulation
from sas_rmc.shapes.cube import Cube


rng = constants.RNG
PI = constants.PI


def different_random_int(n: int, number_to_avoid: int) -> int: # raises ValueError
    for _ in range(200000):
        x = random.choice(range(n))
        if x != number_to_avoid:
            return x
    raise ValueError(f"It is impossible to avoid {number_to_avoid} from a range of {n}")

def embiggen_core_shell_particle(particle: Particle, embiggen_factor: float) -> Particle:
    if isinstance(particle, CoreShellParticle):
        return CoreShellParticle.gen_from_parameters(
            position=particle.get_position(),
            magnetization=particle.get_magnetization(),
            core_radius=particle.core_radius * embiggen_factor,
            thickness=particle.thickness,
            core_sld=particle.core_sld,
            shell_sld=particle.shell_sld,
            solvent_sld=particle.solvent_sld
        )
    raise TypeError()

def embiggen_core_shell_shell(particle: Particle, embiggen_factor: float) -> Particle:
    if isinstance(particle, CoreShellParticle):
        return CoreShellParticle.gen_from_parameters(
            position=particle.get_position(),
            magnetization=particle.get_magnetization(),
            core_radius=particle.core_radius,
            thickness=particle.thickness * embiggen_factor,
            core_sld=particle.core_sld,
            shell_sld=particle.shell_sld,
            solvent_sld=particle.solvent_sld
        )
    raise TypeError()

def embiggen_option(particle: Particle, embiggen_factor: float) -> Particle:
    return random.choice([
        embiggen_core_shell_particle(particle, embiggen_factor),
        embiggen_core_shell_shell(particle, embiggen_factor)
    ])

def create_command(
        box_index: int,
        particle_index: int,
        move_by_distance: float,
        cube: Cube,
        total_particle_number: int,
        nominal_angle_change: float = PI/8,
        nominal_rescale_change: float = 0.02,
        nominal_magnetization: float = 0,
        ) -> commands.Command:
    possible_commands = [
        commands.MoveParticleBy(
            box_index=box_index,
            particle_index=particle_index,
            position_delta=Vector.random_vector_xy(length=rng.normal(loc = 0, scale = move_by_distance))
            ),
        commands.MoveParticleTo(
            box_index=box_index,
            particle_index=particle_index,
            position_new=cube.random_position_inside().project_to_xy()
            ),
        commands.JumpParticleTo(
            box_index=box_index,
            particle_index=particle_index,
            reference_particle_index=different_random_int(total_particle_number, particle_index)
            ),
        commands.OrbitParticle(
            box_index=box_index,
            particle_index=particle_index,
            relative_angle=rng.normal(loc = 0, scale=nominal_angle_change),
            ),
        commands.MagnetizeParticle(
            box_index=box_index,
            particle_index=particle_index,
            magnetization=Vector.random_vector_xy(length=rng.normal(loc = 0, scale = nominal_magnetization))
            ),
        commands.RescaleMagnetization(
            box_index=box_index,
            particle_index=particle_index,
            rescale_factor=rng.normal(loc = 1.0, scale=nominal_rescale_change)
            ),
        commands.RotateMagnetization(
           box_index=box_index,
            particle_index=particle_index,
            relative_angle=rng.normal(loc = 0, scale=nominal_angle_change),
            ), 
        commands.FlipMagnetization(
            box_index=box_index,
            particle_index=particle_index
            ),
        commands.RelativeRescale(
            change_by_factor=rng.normal(loc = 1.0, scale=nominal_rescale_change)
            ),
        commands.MutateParticle(
            box_index=box_index,
            particle_index=particle_index,
            particle_mutation_function=lambda p : embiggen_option(p, rng.normal(loc = 1.0, scale=nominal_rescale_change))
            ),
        ]
    return random.choice(possible_commands)


@pydantic_dataclass
class CommandFactory:
    allow_particle_physics: bool = True
    allow_magnetization_changes: bool = True
    allow_morphology_changes: bool = True
    allow_simulation_changes: bool = True
    nominal_angle_change: float = PI/8
    nominal_rescale_change: float = 0.02
    nominal_magnetization: float = 0
    move_by_distance: float | None = None

    def create_move_by(self, simulation_state: ScatteringSimulation, box_index: int, particle_index: int) -> commands.MoveParticleBy:
        move_by_distance = self.move_by_distance if self.move_by_distance is not None else (simulation_state.get_particle(box_index, particle_index).get_volume()**(1/3)) / 2
        return commands.MoveParticleBy(
            box_index=box_index,
            particle_index=particle_index,
            position_delta=Vector.random_vector_xy(length=rng.normal(loc = 0, scale = move_by_distance))
        )
    
    def create_move_to(self, simulation_state: ScatteringSimulation, box_index: int, particle_index: int) -> commands.MoveParticleTo:
        return commands.MoveParticleTo(
            box_index=box_index,
            particle_index=particle_index,
            position_new=simulation_state.box_list[box_index].cube.random_position_inside().project_to_xy()
        )
    
    def create_jump_particle_to(self, simulation_state: ScatteringSimulation, box_index: int, particle_index: int) -> commands.JumpParticleTo:
        return commands.JumpParticleTo(
            box_index=box_index,
            particle_index=particle_index,
            reference_particle_index=different_random_int(len(simulation_state.box_list[box_index]), particle_index)
        )
    
    def create_orbit_particle(self, _: ScatteringSimulation, box_index: int, particle_index: int) -> commands.OrbitParticle:
        return commands.OrbitParticle(
            box_index=box_index,
            particle_index=particle_index,
            relative_angle=rng.normal(loc = 0, scale=self.nominal_angle_change),
        )
    
    def create_magnetize_particle(self, _: ScatteringSimulation, box_index: int, particle_index: int) -> commands.MagnetizeParticle:
        return commands.MagnetizeParticle(
            box_index=box_index,
            particle_index=particle_index,
            magnetization=Vector.random_vector_xy(length=rng.normal(loc = 0, scale = self.nominal_magnetization))
        )
        
    def create_rescale_magnetization(self, _: ScatteringSimulation, box_index: int, particle_index: int) -> commands.RescaleMagnetization:
        return commands.RescaleMagnetization(
            box_index=box_index,
            particle_index=particle_index,
            rescale_factor=rng.normal(loc = 1.0, scale=self.nominal_rescale_change)
        )
    
    def create_flip_magnetization(self, _: ScatteringSimulation, box_index: int, particle_index: int) -> commands.FlipMagnetization:
        return commands.FlipMagnetization(
            box_index=box_index,
            particle_index=particle_index
        )
    
    def create_relative_rescale(self, _: ScatteringSimulation, __: int, ___: int) -> commands.RelativeRescale:
        return commands.RelativeRescale(
            change_by_factor=rng.normal(loc = 1.0, scale=self.nominal_rescale_change)
        )
    
    def create_embiggen_core_shell_particle(self, _: ScatteringSimulation, box_index: int, particle_index: int) -> commands.MutateParticle:
        return commands.MutateParticle(
            box_index=box_index,
            particle_index=particle_index,
            particle_mutation_function=lambda p : embiggen_core_shell_particle(p, embiggen_factor=rng.normal(loc = 1.0, scale=self.nominal_rescale_change))
        )
    
    def create_embiggen_core_shell_shell(self, _: ScatteringSimulation, box_index: int, particle_index: int) -> commands.MutateParticle:
        return commands.MutateParticle(
            box_index=box_index,
            particle_index=particle_index,
            particle_mutation_function=lambda p : embiggen_core_shell_shell(p, embiggen_factor=rng.normal(loc = 1.0, scale=self.nominal_rescale_change))
        )
        
    def create_command(self, simulation_state: ScatteringSimulation, box_index: int, particle_index: int) -> commands.Command:
        create_allowed_commands = []
        if self.allow_particle_physics:
            create_allowed_commands.extend([
                self.create_move_by,
                self.create_move_to,
                self.create_jump_particle_to,
                self.create_orbit_particle
            ])
        if self.allow_magnetization_changes and self.nominal_magnetization:
            create_allowed_commands.extend([
                self.create_magnetize_particle,
                self.create_rescale_magnetization,
                self.create_flip_magnetization
            ])
        if self.allow_simulation_changes:
            create_allowed_commands.extend([
                self.create_relative_rescale
            ])
        if self.allow_morphology_changes:
            if isinstance(simulation_state.get_particle(box_index, particle_index), CoreShellParticle):
                create_allowed_commands.extend([
                    self.create_embiggen_core_shell_particle,
                    self.create_embiggen_core_shell_shell
                ])
        create = random.choice(create_allowed_commands)
        return create(simulation_state, box_index, particle_index)

    @classmethod
    def create_from_dataframes(cls, dataframes: dict[str, pd.DataFrame]):
        value_frame = parse_data.parse_value_frame(dataframes['Simulation parameters'])
        return cls(**value_frame)


if __name__ == "__main__":
    print(different_random_int(100,0))

    
# %%
