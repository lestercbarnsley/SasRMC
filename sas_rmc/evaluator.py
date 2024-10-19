#%%
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from sas_rmc import constants
from sas_rmc.array_cache import method_array_cache
from sas_rmc.acceptance_scheme import AcceptanceScheme
from sas_rmc.detector import DetectorImage, make_smearing_function, DEFAULT_GAUSSIAN_FLOOR_FRACTION
from sas_rmc.result_calculator import ResultCalculator
from sas_rmc.scattering_simulation import ScatteringSimulation


PI = constants.PI


@dataclass
class Evaluator(ABC):

    @abstractmethod
    def evaluate_and_get_document(self, simulation_state: ScatteringSimulation, acceptance_scheme: AcceptanceScheme) -> tuple[bool, dict]:
        pass

    @abstractmethod
    def default_box_dimensions(self) -> list[float]:
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
    def default_box_dimensions(self) -> list[float]:
        pass

    @abstractmethod
    def get_loggable_data(self, simulation_state: ScatteringSimulation) -> dict:
        pass


def calculate_goodness_of_fit(simulated_intensity: np.ndarray, experimental_detector: DetectorImage) -> float:
    experimental_intensity = experimental_detector.intensity
    uncertainty = experimental_detector.intensity_err
    return ((experimental_intensity - simulated_intensity)**2 / uncertainty**2)[experimental_detector.shadow_factor].mean()

def qXqY_delta(detector: DetectorImage) -> tuple[float, float]:
    qXs = np.unique(detector.qX)
    qYs = np.unique(detector.qY)
    qX_diff = np.max(np.gradient(qXs))
    qY_diff = np.max(np.gradient(qYs))
    return float(qX_diff), float(qY_diff)
    

@dataclass
class Smearing2DFitter(Fitter):
    result_calculator: ResultCalculator
    experimental_detector: DetectorImage
    qx_matrix: np.ndarray
    qy_matrix: np.ndarray
    gaussian_floor: float = DEFAULT_GAUSSIAN_FLOOR_FRACTION

    @method_array_cache
    def create_smearing_function(self) -> Callable[[np.ndarray], np.ndarray]:
        return make_smearing_function(
            self.experimental_detector.detector_pixels,
            qx_matrix=self.qx_matrix,
            qy_matrix=self.qy_matrix,
            gaussian_floor=self.gaussian_floor
        )
    
    def default_box_dimensions(self) -> list[float]:
        qX_diff, qY_diff = qXqY_delta(self.experimental_detector)
        return [2 * PI / qX_diff, 2 * PI / qY_diff, 2 * PI / qX_diff]
        
    def simulate_intensity(self, simulation_state: ScatteringSimulation) -> np.ndarray:
        intensity_result = self.result_calculator.intensity_result(simulation_state)
        smearing_function = self.create_smearing_function()
        return smearing_function(intensity_result)

    def calculate_goodness_of_fit(self, simulation_state: ScatteringSimulation) -> float:
        simulated_intensity = self.simulate_intensity(simulation_state)
        return calculate_goodness_of_fit(simulated_intensity, self.experimental_detector)

    def get_loggable_data(self, simulation_state: ScatteringSimulation) -> dict:
        return {
            "Result calculator" : type(self.result_calculator).__name__,
            "Detector data" : self.experimental_detector.get_loggable_data(),
            "Simulated intensity" : [float(intensity) for intensity in self.simulate_intensity(simulation_state)],
            "Polarization" : self.experimental_detector.polarization.value,
            "Smearing" : True
        }
    
@dataclass
class NoSmearing2DFitter(Fitter):
    result_calculator: ResultCalculator
    experimental_detector: DetectorImage

    def simulate_intensity(self, simulation_state: ScatteringSimulation) -> np.ndarray:
        intensity_result = self.result_calculator.intensity_result(simulation_state)
        return intensity_result

    def default_box_dimensions(self) -> list[float]:
        qX_diff, qY_diff = qXqY_delta(self.experimental_detector)
        return [2 * PI / qX_diff, 2 * PI / qY_diff, 2 * PI / qX_diff]

    def calculate_goodness_of_fit(self, simulation_state: ScatteringSimulation) -> float:
        simulated_intensity = self.simulate_intensity(simulation_state)
        return calculate_goodness_of_fit(simulated_intensity, self.experimental_detector)

    def get_loggable_data(self, simulation_state: ScatteringSimulation) -> dict:
        return {
            "Result calculator" : type(self.result_calculator).__name__,
            "Detector data" : self.experimental_detector.get_loggable_data(),
            "Simulated intensity" : [float(intensity) for intensity in self.simulate_intensity(simulation_state)],
            "Polarization" : self.experimental_detector.polarization.value,
            "Smearing" : False
        }

@dataclass
class FitterMultiple(Fitter):
    fitter_list: list[Fitter]
    weight: list[float] | None = None

    def calculate_goodness_of_fit(self, simulation_state: ScatteringSimulation) -> float:
        return float(
            np.average(
                [fitter.calculate_goodness_of_fit(simulation_state) for fitter in self.fitter_list], 
                weights=self.weight)
            )

    def default_box_dimensions(self) -> list[float]:
        return max(
            [fitter.default_box_dimensions() for fitter in self.fitter_list], 
            key=lambda box_dimension : float(np.prod(box_dimension))
            )

    def get_loggable_data(self, simulation_state: ScatteringSimulation) -> dict:
        return {f"Fitter {i}" : fitter.get_loggable_data(simulation_state) 
                for i, fitter 
                in enumerate(self.fitter_list)}

@dataclass
class EvaluatorWithFitter(Evaluator):
    fitter: Fitter
    current_chi_squared: float = np.inf

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
    
    def default_box_dimensions(self) -> list[float]:
        return self.fitter.default_box_dimensions()
    
    def get_loggable_data(self, simulation_state: ScatteringSimulation) -> dict:
        return {
            "Evaluator" : type(self).__name__,
            "Current goodness of fit" : self.current_chi_squared,
            "Fitter" : self.fitter.get_loggable_data(simulation_state)
            }


if __name__ == "__main__":
    pass


# %%
