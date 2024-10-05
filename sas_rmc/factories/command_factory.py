#%%
import random

from sas_rmc import commands, Vector, constants
from sas_rmc.shapes.cube import Cube


rng = constants.RNG
PI = constants.PI


def different_random_int(n: int, number_to_avoid: int) -> int: # raises ValueError
    for _ in range(200000):
        x = random.choice(range(n))
        if x != number_to_avoid:
            return x
    raise ValueError(f"It is impossible to avoid {number_to_avoid} from a range of {n}")

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
            position_new=cube.random_position_inside()
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
        commands.RescaleBoxMagnetization(
            box_index=box_index,
            particle_index=particle_index,
            rescale_factor=rng.normal(loc = 1.0, scale=nominal_rescale_change)
            ),
        commands.RelativeRescale(
            change_by_factor=rng.normal(loc = 1.0, scale=nominal_rescale_change)
            ),
        
        ]
    return random.choice(possible_commands)



if __name__ == "__main__":
    print(different_random_int(100,0))
# %%
