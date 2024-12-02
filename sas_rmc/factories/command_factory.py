#%%
import random

from sas_rmc import commands, Vector, constants
from sas_rmc.particles import Particle, ParticleArray
from sas_rmc.particles.particle_core_shell_spherical import CoreShellParticle
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



if __name__ == "__main__":
    print(different_random_int(100,0))
# %%
