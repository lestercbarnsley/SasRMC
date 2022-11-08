#%%
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable, List

import numpy as np

from ..form_calculator import FieldDirection
from ..result_calculator import ResultCalculator
from ..detector import DetectorImage
from ..box_simulation import Box
from ..scattering_simulation import ScatteringSimulation, SimulationParam, SimulationParams, NUCLEAR_RESCALE, MAGNETIC_RESCALE
from ..fitter import Fitter2D


@dataclass
class SimulationFactory(ABC):

    @abstractmethod
    def create_simulation(self, detector_list: List[DetectorImage], box_list: List[Box]) -> ScatteringSimulation:
        pass


def box_simulation_params_factory(starting_rescale: float = 1.0, starting_magnetic_rescale: float = 1.0) -> SimulationParams:
    params = [
        SimulationParam(value = starting_rescale, name = NUCLEAR_RESCALE, bounds=(0, np.inf)), 
        SimulationParam(value = starting_magnetic_rescale, name = MAGNETIC_RESCALE, bounds=(0, np.inf))
        ]
    return SimulationParams(params = params)


@dataclass
class Fitter2DSimulationFactory(SimulationFactory):
    detector_smearing: bool
    result_calculator_maker: Callable[[DetectorImage], ResultCalculator]
    nominal_concentration: float
    field_direction: FieldDirection = FieldDirection.Y

    def create_simulation(self, detector_list: List[DetectorImage], box_list: List[Box]) -> ScatteringSimulation:
        fitter = Fitter2D.generate_standard_fitter(
            detector_list=detector_list,
            box_list = box_list,
            result_calculator_maker=self.result_calculator_maker,
            smearing=self.detector_smearing,
            field_direction=self.field_direction
        )
        box_concentration = np.sum([p.volume for p in box_list[0]]) / box_list[0].volume
        starting_factor = self.nominal_concentration / box_concentration if self.nominal_concentration else 1.0
        return ScatteringSimulation(fitter = fitter, simulation_params=box_simulation_params_factory(starting_rescale=starting_factor, starting_magnetic_rescale=starting_factor))


def gen_from_dict(d: dict, result_calculator_maker: Callable[[DetectorImage], ResultCalculator]) -> SimulationFactory: # For now this has just one output, but in the future, it will have other options
    detector_smearing = d.get("detector_smearing")
    nominal_concentration = d.get("nominal_concentration", 0.0)
    field_direction = {
        "X": FieldDirection.X,
        "Y": FieldDirection.Y,
        "Z": FieldDirection.Z
    }[d.get("field_direction", "Y")]
    return Fitter2DSimulationFactory(detector_smearing, result_calculator_maker, nominal_concentration, field_direction)


        
