#%%
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from sas_rmc.constants import np_max, np_average, iter_np_array
from sas_rmc.detector import DetectorImage, make_smearing_function, DEFAULT_GAUSSIAN_FLOOR_FRACTION
from sas_rmc.result_calculator import ProfileCalculator, ResultCalculator
from sas_rmc.array_cache import method_array_cache
from sas_rmc.acceptance_scheme import AcceptanceScheme
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
class Fitter(ABC):

    @abstractmethod
    def calculate_goodness_of_fit(self, simulation_state: ScatteringSimulation) -> float:
        pass

    @abstractmethod
    def get_loggable_data(self, simulation_state: ScatteringSimulation) -> dict:
        pass


def calculate_goodness_of_fit(simulated_intensity: np.ndarray, experimental_detector: DetectorImage) -> float:
    experimental_intensity = experimental_detector.intensity
    uncertainty = np.where(
        experimental_detector.intensity_err == 0, 
        1, 
        experimental_detector.intensity_err
        )
    allowed_idxs = experimental_detector.shadow_factor.nonzero()
    difference_of_squares =((experimental_intensity - simulated_intensity)**2 / uncertainty**2)[allowed_idxs]
    return np.average(difference_of_squares).item() # Use np.average here because the constant function takes a list

def qXqY_delta(detector: DetectorImage) -> tuple[float, float]:
    qXs = np.unique(detector.qX)
    qYs = np.unique(detector.qY)
    qX_diff = np_max(np.gradient(qXs))
    qY_diff = np_max(np.gradient(qYs))
    return qX_diff, qY_diff
    

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
            "Simulated intensity" : [intensity for intensity in iter_np_array(self.simulate_intensity(simulation_state))],
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

    def calculate_goodness_of_fit(self, simulation_state: ScatteringSimulation) -> float:
        simulated_intensity = self.simulate_intensity(simulation_state)
        return calculate_goodness_of_fit(simulated_intensity, self.experimental_detector)

    def get_loggable_data(self, simulation_state: ScatteringSimulation) -> dict:
        return {
            "Result calculator" : type(self.result_calculator).__name__,
            "Detector data" : self.experimental_detector.get_loggable_data(),
            "Simulated intensity" : [intensity for intensity in iter_np_array(self.simulate_intensity(simulation_state))],
            "Polarization" : self.experimental_detector.polarization.value,
            "Smearing" : False
        }


@dataclass
class FitterMultiple(Fitter):
    fitter_list: list[Fitter]
    weight: list[float] | None = None

    def calculate_goodness_of_fit(self, simulation_state: ScatteringSimulation) -> float:
        return np_average(
            [fitter.calculate_goodness_of_fit(simulation_state) for fitter in self.fitter_list], 
            weights=self.weight
        )

    def get_loggable_data(self, simulation_state: ScatteringSimulation) -> dict:
        return {f"Fitter {i}" : fitter.get_loggable_data(simulation_state) 
                for i, fitter 
                in enumerate(self.fitter_list)}
    

@dataclass
class ProfileFitter(Fitter):
    profile_calculator: ProfileCalculator
    experimental_intensity: np.ndarray
    experimental_uncertainty: np.ndarray | None = None

    def simulate_intensity(self, simulation_state: ScatteringSimulation) -> np.ndarray:
        return self.profile_calculator.intensity_result(simulation_state)

    def calculate_goodness_of_fit(self, simulation_state: ScatteringSimulation) -> float:
        simulated_intensity = self.simulate_intensity(simulation_state)
        uncertainty = self.experimental_uncertainty if self.experimental_uncertainty is not None else np.ones(self.experimental_intensity.shape)
        difference_of_squares = ((self.experimental_intensity - simulated_intensity)**2 / uncertainty**2)
        return np.average(difference_of_squares).item()
    
    def get_loggable_data(self, simulation_state: ScatteringSimulation) -> dict:
        experimental_uncertainty = self.experimental_uncertainty if self.experimental_uncertainty is not None else np.zeros(self.experimental_intensity.shape)
        return {
            "Result calculator" : type(self.profile_calculator).__name__,
            "Experimental intensity" : [exp_intensity for exp_intensity in iter_np_array(self.experimental_intensity)],
            "Experimental uncertainty" : [exp_intensity for exp_intensity in iter_np_array(experimental_uncertainty)],
            "Simulated intensity" : [intensity for intensity in iter_np_array(self.simulate_intensity(simulation_state))],
        }

    

def get_evaluation_document(evaluator_name: str, current_goodness_of_fit: float, evaluation: bool, md: dict | None = None) -> dict:
    md = md if md is not None else {}
    return {
        "Evaluator" : evaluator_name,
        "Current goodness of fit" : current_goodness_of_fit,
        "Acceptance" : evaluation
    } | md


@dataclass
class EvaluatorWithFitter(Evaluator):
    fitter: Fitter
    current_chi_squared: float = np.inf

    def evaluation_without_physical_acceptance(self, acceptance_scheme: AcceptanceScheme) -> tuple[bool, dict]:
        physical_md = {"Physical acceptance" : False}
        document = get_evaluation_document(
            evaluator_name=type(self).__name__,
            current_goodness_of_fit=self.current_chi_squared,
            evaluation=False,
            md = acceptance_scheme.get_loggable_data() | physical_md
        )
        return False, document
    
    def evaluation_with_success(self, acceptance_scheme: AcceptanceScheme, new_goodness_of_fit: float) -> tuple[bool, dict]:
        physical_md = {"Physical acceptance" : True}
        document = get_evaluation_document(
            evaluator_name=type(self).__name__,
            current_goodness_of_fit=new_goodness_of_fit,
            evaluation=True,
            md = acceptance_scheme.get_loggable_data() | physical_md
        )
        return True, document
    
    def evaluation_without_success(self, acceptance_scheme: AcceptanceScheme, new_goodness_of_fit: float) -> tuple[bool, dict]:
        physical_md = {"Physical acceptance" : True}
        document = get_evaluation_document(
            evaluator_name=type(self).__name__,
            current_goodness_of_fit=new_goodness_of_fit,
            evaluation=False,
            md = acceptance_scheme.get_loggable_data() | physical_md
        )
        return False, document
    
    def update_chi_squared(self, new_chi_squared: float) -> None:
        self.current_chi_squared = new_chi_squared

    def evaluate_and_get_document(self, simulation_state: ScatteringSimulation, acceptance_scheme: AcceptanceScheme) -> tuple[bool, dict]:
        if not simulation_state.get_physical_acceptance():
            return self.evaluation_without_physical_acceptance(acceptance_scheme)
        new_goodness_of_fit = self.fitter.calculate_goodness_of_fit(simulation_state)
        if acceptance_scheme.is_acceptable(self.current_chi_squared, new_goodness_of_fit):
            self.update_chi_squared(new_chi_squared=new_goodness_of_fit) # State has changed
            return self.evaluation_with_success(acceptance_scheme, self.current_chi_squared)
        return self.evaluation_without_success(acceptance_scheme, new_goodness_of_fit)   
    
    def get_loggable_data(self, simulation_state: ScatteringSimulation) -> dict:
        return {
            "Evaluator" : type(self).__name__,
            "Current goodness of fit" : self.current_chi_squared,
            "Fitter" : self.fitter.get_loggable_data(simulation_state)
            }


if __name__ == "__main__":
    pass



# %%
