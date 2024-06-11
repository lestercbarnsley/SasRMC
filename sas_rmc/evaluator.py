#%%
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from sas_rmc.acceptance_scheme import AcceptanceScheme
from sas_rmc.detector import DetectorImage, make_smearing_function
from sas_rmc.result_calculator import AnalyticalCalculator
from sas_rmc.scattering_simulation import ScatteringSimulation


@dataclass
class Evaluator(ABC):

    @abstractmethod
    def evaluate_and_get_document(self, simulation_state: ScatteringSimulation, acceptance_scheme: AcceptanceScheme) -> tuple[bool, dict]:
        pass

    @abstractmethod
    def get_loggable_data(self, simulation_state: ScatteringSimulation) -> dict:
        pass


def get_evaluation_document(evaluator_name: str, current_goodness_of_fit: float, evaluation: bool, md: dict | None = None) -> dict:
    md = md if md is not None else {}
    return {
        "Evaluator" : evaluator_name,
        "Current goodness of fit" : current_goodness_of_fit,
        "Acceptance" : evaluation
    } | md


@dataclass
class Fitter(ABC):

    @abstractmethod
    def calculate_goodness_of_fit(self, simulation_state: ScatteringSimulation) -> float:
        pass

    @abstractmethod
    def get_loggable_data(self, simulation_state: ScatteringSimulation) -> dict:
        pass


@dataclass
class Smearing2DFitter(Fitter):
    result_calculator: AnalyticalCalculator
    experimental_detector: DetectorImage
    smearing_function: Callable[[np.ndarray], np.ndarray] | None = None

    def create_smearing_function(self) -> Callable[[np.ndarray], np.ndarray]:
        return make_smearing_function(
            self.experimental_detector.detector_pixels,
            qx_matrix=self.result_calculator.qx_array,
            qy_matrix=self.result_calculator.qy_array
        )
    
    def simulate_intensity(self, simulation_state: ScatteringSimulation) -> np.ndarray:
        intensity_result = self.result_calculator.intensity_result(simulation_state)
        if self.smearing_function is None:
            self.smearing_function = self.create_smearing_function()
        return self.smearing_function(intensity_result)

    def calculate_goodness_of_fit(self, simulation_state: ScatteringSimulation) -> float:
        simulated_intensity = self.simulate_intensity(simulation_state)
        experimental_intensity = self.experimental_detector.intensity
        uncertainty = self.experimental_detector.intensity_err
        return ((experimental_intensity - simulated_intensity)**2 / uncertainty**2)[self.experimental_detector.shadow_factor].mean()
    
    def get_loggable_data(self, simulation_state: ScatteringSimulation) -> dict:
        return {
            "Result calculator" : type(self.result_calculator).__name__,
            "Detector data" : self.experimental_detector.get_loggable_data(),
            "Simulated intensity" : [intensity for intensity in self.simulate_intensity(simulation_state)],
            "Smearing" : True
        }

@dataclass
class Smearing2dFitterMultiple(Fitter):
    fitter_list: list[Smearing2DFitter]

    def calculate_goodness_of_fit(self, simulation_state: ScatteringSimulation) -> float:
        return np.sum([fitter.calculate_goodness_of_fit(simulation_state) for fitter in self.fitter_list])

    def get_loggable_data(self, simulation_state: ScatteringSimulation) -> dict:
        return {f"Fitter {i}" : fitter.get_loggable_data(simulation_state) 
                for i, fitter 
                in enumerate(self.fitter_list)}

@dataclass
class EvaluatorWithFitter(Evaluator):
    current_chi_squared: float
    fitter: Fitter

    def evaluate_and_get_document(self, simulation_state: ScatteringSimulation, acceptance_scheme: AcceptanceScheme) -> tuple[bool, dict]:
        if not simulation_state.get_physical_acceptance():
            document = get_evaluation_document(
                evaluator_name=type(self).__name__,
                current_goodness_of_fit=self.current_chi_squared,
                evaluation=False,
                md = acceptance_scheme.get_loggable_data() | {"Physical acceptance" : False}
            )
            return False, document
        md = {"Physical acceptance" : True}
        new_goodness_of_fit = self.fitter.calculate_goodness_of_fit(simulation_state)
        if acceptance_scheme.is_acceptable(self.current_chi_squared, new_goodness_of_fit):
            self.current_chi_squared = new_goodness_of_fit
            document = get_evaluation_document(
                evaluator_name=type(self).__name__,
                current_goodness_of_fit=new_goodness_of_fit,
                evaluation=True,
                md = acceptance_scheme.get_loggable_data() | md
            )
            return True, document
        document = get_evaluation_document(
            evaluator_name=type(self).__name__,
            current_goodness_of_fit=new_goodness_of_fit,
            evaluation=False,
            md = acceptance_scheme.get_loggable_data() | md
        )
        return False, document
    
    def get_loggable_data(self, simulation_state: ScatteringSimulation) -> dict:
        return {
            "Evaluator" : type(self).__name__,
            "Current goodness of fit" : self.current_chi_squared,
            "Fitter" : self.fitter.get_loggable_data(simulation_state)
            }


if __name__ == "__main__":
    pass


# %%
