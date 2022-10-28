#%%

from .shapes import shapes
from .particles import particle
from . import commands, scattering_simulation, simulator, acceptance_scheme, viewer, array_cache, simulator_factory, result_calculator, form_calculator

#from .rmc_runner import RmcRunner, load_config
from .factories.runner_factory import load_config

# Only very import classes should be directly imported
# This is just for putting things into the namespace
from .particles.particle import Particle
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