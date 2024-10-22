#%%

from sas_rmc import Vector, constants
from sas_rmc.particles.particle_core_shell_spherical import CoreShellParticleForm
from sas_rmc.factories.parse_data import coerce_types

rng = constants.RNG


def polydisperse_parameter(loc: float, polyd: float) -> float:
    return rng.normal(loc = loc, scale = loc * polyd)

@coerce_types
def create_core_shell_particle(
                core_radius: float,
                core_polydispersity: float,
                core_sld: float,
                shell_thickness: float,
                shell_polydispersity: float,
                shell_sld: float,
                solvent_sld: float,
                core_magnetization: float,
                ) -> CoreShellParticleForm:
    return CoreShellParticleForm.gen_from_parameters(
        position=Vector.null_vector(),
        magnetization=core_magnetization * Vector.random_vector_xy(),
        core_radius=polydisperse_parameter(core_radius, core_polydispersity),
        thickness=polydisperse_parameter(shell_thickness, shell_polydispersity),
        core_sld=core_sld,
        shell_sld=shell_sld,
        solvent_sld=solvent_sld
    )
    


#%%
