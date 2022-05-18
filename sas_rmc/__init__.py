#%%

from . import commands, particle, scattering_simulation, simulator, acceptance_scheme, viewer, shapes, array_cache, simulator_factory

from .rmc_runner import RmcRunner, load_config

# Only very import classes should be directly imported
from .particle import Particle
from .vector import Vector, VectorSpace, VectorElement
from .detector import DetectorImage, SimulatedDetectorImage, DetectorPixel, DetectorConfig, Polarization
from .controller import Controller
from .logger import Logger

'''def main():
    rmc_runner = load_config("../../SasRMC/data/config.yaml")
    rmc_runner.run()'''

if __name__ == "__main__":
    pass#main()
    
# %%