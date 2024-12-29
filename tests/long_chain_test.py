#%%

import numpy as np

from sas_rmc.box_simulation import Box
from sas_rmc.particles.particle_spherical import SphericalParticleProfile
from sas_rmc.scattering_simulation import ScatteringSimulation, SimulationParam


def main() -> None:
    pass

def create_box_list() -> list[Box]:
    return [
        Box(particle_results=[SphericalParticleProfile()])
    ]
    

def create_simulation() -> ScatteringSimulation:
    return ScatteringSimulation(
        scale_factor=SimulationParam(value=1.0, name="scale_factor", bounds=(0, np.inf)),
        box_list=
    )

if __name__ == "__main__":
    main()

