'''import pytest
from pathlib import Path

from typing import List

import numpy as np
import sas_rmc

from sas_rmc.box_simulation import Box
from sas_rmc.controller import Controller
from sas_rmc.particle import CoreShellParticle, Dumbbell, Particle
from sas_rmc.scattering_simulation import MAGNETIC_RESCALE, NUCLEAR_RESCALE, ScatteringSimulation
from sas_rmc import Vector, SimulatedDetectorImage, Polarization, DetectorConfig, commands, shapes
from sas_rmc.fitter import Fitter2D


PI = np.pi

'''