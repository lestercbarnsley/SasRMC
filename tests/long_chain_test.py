#%%

import numpy as np
from matplotlib import pyplot as plt

from sas_rmc.box_simulation import Box, Cube
from sas_rmc.particles.particle_spherical import SphericalParticleProfile
from sas_rmc.result_calculator import ProfileCalculator
from sas_rmc.scattering_simulation import ScatteringSimulation, SimulationParam
from sas_rmc.vector import Vector
from sas_rmc.shapes import Cylinder


def create_spherical_particle_profile(position: Vector, sphere_radius: float) -> SphericalParticleProfile:
    return SphericalParticleProfile.gen_from_parameters(
        position=position,
        sphere_radius=sphere_radius,
        sphere_sld=6.9,
        solvent_sld=0
    )

def create_cylinder(central_position: Vector, radius: float, height: float) -> list[SphericalParticleProfile]:
    SPHERE_RADIUS = 20

    cylinder = Cylinder(radius=radius, height=height, central_position=central_position, orientation=Vector(0,1))

    particles = []
    for x in np.arange(start=-radius, stop=+radius, step = SPHERE_RADIUS * 2):
        for z in np.arange(start= -radius, stop= + radius, step=SPHERE_RADIUS):
            for y in np.arange(start = -height/2, stop=+height/2 , step = SPHERE_RADIUS):
                position = central_position + Vector(x, y, z)
                if cylinder.is_inside(position):
                    particles.append(
                        SphericalParticleProfile.gen_from_parameters(
                            position=position,
                            sphere_radius=SPHERE_RADIUS,
                            sphere_sld=6.9
                        )
                    )
    return particles

def create_box_list() -> list[Box]:
    radius = 40
    height = 1000
    '''return [
        Box(
            particle_results=[create_spherical_particle_profile(Vector(0, 2 * i * radius, 0), 100) for i in range(int(height / radius))],
            cube = Cube(Vector.null_vector(), orientation=Vector(0,1, 0), dimension_0=100_000, dimension_1=100_000, dimension_2=100_000))      
    ]'''
    return [
        Box(
            particle_results=create_cylinder(Vector.null_vector(), radius=radius, height=height),
            cube=Cube(Vector.null_vector(), Vector(0,1), 100_000, 100_000, 100_000)
        )
    ]
    

def create_simulation() -> ScatteringSimulation:
    return ScatteringSimulation(
        scale_factor=SimulationParam(value=1.0, name="scale_factor", bounds=(0, np.inf)),
        box_list=create_box_list()
    ).validate_state()

def create_result_calculator() -> ProfileCalculator:
    return ProfileCalculator(
        q_profile=np.linspace(2e-3, 0.2, num = 100)
    )

def main() -> None:
    simulation = create_simulation()
    profile_calculator = create_result_calculator()
    res = profile_calculator.intensity_result(simulation)

    plt.loglog(profile_calculator.q_profile, res)
    plt.loglog(profile_calculator.q_profile, 0.01* profile_calculator.q_profile**(-1))
    plt.show()

    box = simulation.box_list[0]

    plt.plot(
        [particle.get_particle().get_position().x for particle in box.particle_results],
        [particle.get_particle().get_position().y for particle in box.particle_results],
        'b.'
        )
    plt.show()




if __name__ == "__main__":
    main()

#%%
