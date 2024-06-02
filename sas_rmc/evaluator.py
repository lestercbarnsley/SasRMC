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
    def evaluate(self, simulation_state: ScatteringSimulation, acceptance_scheme: AcceptanceScheme) -> bool:
        pass

    @abstractmethod
    def get_document(self) -> dict:
        pass

    @abstractmethod
    def get_loggable_data(self) -> dict:
        pass


@dataclass
class Evaluator2DSmearing(Evaluator):
    current_chi_squared: float
    result_calculator: AnalyticalCalculator
    experimental_detector: DetectorImage
    smearing_function: Callable[[np.ndarray], np.ndarray] | None = None
    current_document: dict | None = None

    def get_document(self) -> dict:
        return self.current_document if self.current_document is not None else {}

    def create_smearing_function(self) -> Callable[[np.ndarray], np.ndarray]:
        return make_smearing_function(
            self.experimental_detector.detector_pixels,
            qx_matrix=self.result_calculator.qx_array,
            qy_matrix=self.result_calculator.qy_array
        )

    def calculate_goodness_of_fit(self, simulated_intensity: np.ndarray) -> float:
        experimental_intensity = self.experimental_detector.intensity
        uncertainty = self.experimental_detector.intensity_err
        return np.sum((experimental_intensity - simulated_intensity)**2 / uncertainty**2)
    
    def set_document(self, acceptance_scheme: AcceptanceScheme, evaluation: bool, md: dict | None = None) -> None:
        md = md if md is not None else {}
        self.current_document = {
            "Evaluator" : type(self).__name__,
            "Current goodness of fit" : self.current_chi_squared,
            "Acceptance" : evaluation
        } | acceptance_scheme.get_loggable_data() | md

    def evaluate(self, simulation_state: ScatteringSimulation, acceptance_scheme: AcceptanceScheme) -> bool:
        if not simulation_state.get_physical_acceptance():
            self.set_document(acceptance_scheme, False, {"Physical acceptance" : False})
            return False
        intensity_result = self.result_calculator.intensity_result(simulation_state)
        if self.smearing_function is None:
            self.smearing_function = self.create_smearing_function()
        simulated_intensity = self.smearing_function(intensity_result)
        new_goodness_of_fit = self.calculate_goodness_of_fit(simulated_intensity)
        if acceptance_scheme.is_acceptable(old_goodness_of_fit=self.current_chi_squared):
            self.current_chi_squared = new_goodness_of_fit
            self.set_document(acceptance_scheme, True)
            return True
        self.set_document(acceptance_scheme, False)
        return False



# %%
