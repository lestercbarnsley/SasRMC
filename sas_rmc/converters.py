

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from sas_rmc import Vector
from sas_rmc.particles import Dumbbell, Particle, CoreShellParticle




def dict_to_particle(d: dict) -> Particle: # Mark for deletion
    particle_type = d.get('Particle type')
    if particle_type == 'CoreShellParticle':
        return generate_core_shell_particle(d)
    if particle_type == "DumbbellParticle":
        return generate_dumbbell_particle(d)
    if particle_type is None:
        raise KeyError("Particle type is not valid")



def generate_core_shell_particle(d: dict) -> CoreShellParticle:
    return CoreShellParticle.gen_from_parameters(
        position=Vector.from_dict(d, vector_str="Position"),
        magnetization=Vector.from_dict(d, vector_str="Magnetization"),
        core_radius=d.get('Core radius', 0.0),
        thickness=d.get('Shell thickness', 0.0),
        core_sld=d.get('Core SLD', 0.0),
        shell_sld=d.get('Shell SLD', 0.0),
        solvent_sld=d.get('Solvent SLD', 0.0),
    )


def generate_dumbbell_particle(d: dict) -> Dumbbell:
    return Dumbbell.gen_from_parameters(
        core_radius=d.get('Core radius', 0.0),
        seed_radius=d.get('Seed radius', 0.0),
        shell_thickness=d.get('Shell thickness', 0.0),
        core_sld=d.get('Core SLD', 0.0),
        seed_sld=d.get('Seed SLD', 0.0),
        shell_sld=d.get('Shell SLD', 0.0),
        solvent_sld=d.get('Solvent SLD', 0.0),
        position=Vector.from_dict(d, vector_str="Position"),
        orientation=Vector.from_dict(d, vector_str="Orientation"),
        core_magnetization=Vector.from_dict(d, vector_str="MagnetizationCore"),
        seed_magnetization=Vector.from_dict(d, vector_str="MagnetizationSeed"),
    )

def core_shell_to_ax(particle: CoreShellParticle, ax: Axes, inner_colour: str = "blue", outer_colour: str = "black" ) -> None:
    ax.add_patch(particle.shapes[1].get_patches(color = outer_colour))
    ax.add_patch(particle.shapes[0].get_patches(color = inner_colour))
    
def dumbell_to_ax(particle: Dumbbell, ax: Axes, core_colour = "blue", seed_colour = "red") -> None:
    core_shell_to_ax(particle.particle_1, ax, inner_colour=core_colour)
    core_shell_to_ax(particle.particle_2, ax, inner_colour=seed_colour)

def particle_to_axes(particle: Particle, ax: Axes) -> None:
    if isinstance(particle, CoreShellParticle):
        core_shell_to_ax(particle, ax)
    elif isinstance(particle, Dumbbell):
        dumbell_to_ax(particle, ax)
    else:
        ax.add_patch(plt.Circle((particle.position.x, particle.position.y), radius = 100), color = "blue")
