#%%
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


@dataclass
class Evaluator2DSmearing(Evaluator):
    current_chi_squared: float
    result_calculator: AnalyticalCalculator
    experimental_detector: DetectorImage
    smearing_function: Callable[[np.ndarray], np.ndarray] | None = None

    def create_smearing_function(self) -> Callable[[np.ndarray], np.ndarray]:
        return make_smearing_function(
            self.experimental_detector.detector_pixels,
            qx_matrix=self.result_calculator.qx_array,
            qy_matrix=self.result_calculator.qy_array
        )

    def calculate_goodness_of_fit(self, simulated_intensity: np.ndarray) -> float:
        experimental_intensity = self.experimental_detector.intensity
        uncertainty = self.experimental_detector.intensity_err
        return ((experimental_intensity - simulated_intensity)**2 / uncertainty**2)[self.experimental_detector.shadow_factor].mean()
    
    def simulate_intensity(self, simulation_state: ScatteringSimulation) -> np.ndarray:
        intensity_result = self.result_calculator.intensity_result(simulation_state)
        if self.smearing_function is None:
            self.smearing_function = self.create_smearing_function()
        return self.smearing_function(intensity_result)
    
    def get_document(self, acceptance_scheme: AcceptanceScheme, evaluation: bool) -> dict:
        return {
            "Evaluator" : type(self).__name__,
            "Current goodness of fit" : self.current_chi_squared,
            "Acceptance" : evaluation
        } | acceptance_scheme.get_loggable_data()

    def evaluate_and_get_document(self, simulation_state: ScatteringSimulation, acceptance_scheme: AcceptanceScheme) -> tuple[bool, dict]:
        if not simulation_state.get_physical_acceptance():
            return False, self.get_document(acceptance_scheme, False) | {"Physical acceptance" : False}
        md = {"Physical acceptance" : True}
        simulated_intensity = self.simulate_intensity(simulation_state)
        new_goodness_of_fit = self.calculate_goodness_of_fit(simulated_intensity)
        if acceptance_scheme.is_acceptable(old_goodness_of_fit=self.current_chi_squared):
            self.current_chi_squared = new_goodness_of_fit
            return True, self.get_document(acceptance_scheme, True) | md
        return False, self.get_document(acceptance_scheme, False) | md
    
    def get_loggable_data(self, simulation_state: ScatteringSimulation) -> dict:
        simulated_intensity = self.simulate_intensity(simulation_state)
        return {
            "Evaluator" : type(self).__name__,
            "Current goodness of fit" : self.current_chi_squared,
            "Result calculator" : type(self.result_calculator).__name__,
            'experimental_detector' : self.experimental_detector.get_loggable_data(),
            'simulated_intensity' : [intensity for intensity in simulated_intensity],
            'Smearing' : True
            }
    
@dataclass
class Evaluator2DMultiple(Evaluator):
    evaluators: list[Evaluator2DSmearing]
    current_chi_squared: float

    def get_document(self, acceptance_scheme: AcceptanceScheme, evaluation: bool) -> dict:
        return {
            "Evaluator" : type(self).__name__,
            "Current goodness of fit" : self.current_chi_squared,
            "Acceptance" : evaluation
        } | acceptance_scheme.get_loggable_data()

    def evaluate_and_get_document(self, simulation_state: ScatteringSimulation, acceptance_scheme: AcceptanceScheme) -> tuple[bool, dict]:
        if not simulation_state.get_physical_acceptance():
            return False, self.get_document(acceptance_scheme, False) | {"Physical acceptance" : False}
        md = {"Physical acceptance" : True}
        new_goodness_of_fit = np.sum([evaluator.calculate_goodness_of_fit(evaluator.simulate_intensity(simulation_state)) for evaluator in self.evaluators])
        if acceptance_scheme.is_acceptable(old_goodness_of_fit=self.current_chi_squared):
            self.current_chi_squared = new_goodness_of_fit
            return True, self.get_document(acceptance_scheme, True) | md
        return False, self.get_document(acceptance_scheme, False) | md

    def get_loggable_data(self, simulation_state: ScatteringSimulation) -> dict:
        return {f'evaluator_{i}' : evaluator.get_loggable_data(simulation_state) for i, evaluator in enumerate(self.evaluators)}    



# %%
