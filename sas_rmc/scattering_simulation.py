#%%

from dataclasses import dataclass, field
from typing import Callable, List, Protocol, Tuple

import numpy as np

from .box_simulation import Box
from .detector import DetectorImage, Polarization, SimulatedDetectorImage
from .form_calculator import box_intensity_average



@dataclass
class SimulationParams:
    rescale_factor: float = 1
    magnetic_rescale: float = 1

    def get_physical_acceptance(self):
        return self.rescale_factor > 0 and self.magnetic_rescale > 0

IntensityCalculator = Callable[[SimulationParams], Tuple[np.ndarray, np.ndarray, np.ndarray]]

def intensity_calculator(box_list: List[Box], qx_array: np.ndarray, qy_array: np.ndarray, simulation_params: SimulationParams, polarization: Polarization) -> np.ndarray:
    rescale_factor = simulation_params.rescale_factor
    magnetic_rescale = simulation_params.magnetic_rescale
    intensity = box_intensity_average(box_list, qx_array, qy_array, rescale_factor=rescale_factor, magnetic_rescale=magnetic_rescale, polarization=polarization)
    return intensity

def create_intensity_calculator(box_list: List[Box], qx_array, qy_array, detector_image: DetectorImage) -> IntensityCalculator:
    polarization = detector_image.polarization
    return lambda simulation_params : (intensity_calculator(box_list, qx_array, qy_array, simulation_params, polarization), qx_array, qy_array)


class Fitter(Protocol):
    def fit(self, simulation_params: SimulationParams) -> float:
        pass


@dataclass
class ScatteringSimulation:
    fitter: Fitter # The fitter is a strategy class
    simulation_params: SimulationParams = field(default_factory=SimulationParams)
    current_goodness_of_fit: float = np.inf

    def get_goodness_of_fit(self) -> float:
        return self.fitter.fit(self.simulation_params)

    def update_goodness_of_fit(self, new_goodness_of_fit: float) -> None:
        self.current_goodness_of_fit = new_goodness_of_fit

    def get_physical_acceptance(self) -> bool: # Mark for deletion
        return self.simulation_params.get_physical_acceptance()


def default_detector_to_weighting_function(detector: DetectorImage) -> np.ndarray:
    intensity_err = detector.intensity_err
    if np.sum(intensity_err**2) != 0:
        return intensity_err
    else:
        rectified_intensity = np.where(detector.intensity > 0, detector.intensity, np.inf)
        return np.sqrt(rectified_intensity)

def detector_smearer(simulated_intensity_array: np.ndarray, simulated_qx: np.ndarray, simulated_qy: np.ndarray, experimental_intensity: SimulatedDetectorImage) -> SimulatedDetectorImage:
    experimental_intensity.smear(
        intensity = simulated_intensity_array,
        qx_array = simulated_qx,
        qy_array = simulated_qy
    )
    return experimental_intensity

def no_smearer(simulated_intensity_array: np.ndarray, simulated_qx: np.ndarray, simulated_qy: np.ndarray, experimental_intensity: SimulatedDetectorImage) -> SimulatedDetectorImage:
    experimental_intensity.simulated_intensity = simulated_intensity_array
    return experimental_intensity

def chi_squared_fit(experimental_detector: DetectorImage, smeared_detector: SimulatedDetectorImage, weighting_function: Callable[[DetectorImage], np.ndarray]) -> float:
    weight = weighting_function(experimental_detector)
    experimental_intensity = experimental_detector.intensity
    simulated_smeared_intensity = smeared_detector.simulated_intensity
    shadow_factor = experimental_detector.shadow_factor
    chi_squared_getter = lambda : (simulated_smeared_intensity - experimental_intensity)**2 / weight**2
    reduced_array = np.where(shadow_factor, chi_squared_getter(), 0)
    return np.average(reduced_array, weights = shadow_factor)

chi_squared_fit_with_default_uncertainty = lambda experimental_detector, smeared_detector: chi_squared_fit(experimental_detector, smeared_detector, weighting_function=default_detector_to_weighting_function)


@dataclass
class Fitter2D:
    simulated_detectors: List[DetectorImage]
    intensity_calculators: List[IntensityCalculator]
    smear: Callable[[np.ndarray, np.ndarray, np.ndarray, DetectorImage], SimulatedDetectorImage]
    fit_detector: Callable[[DetectorImage, SimulatedDetectorImage], float]

    def _smeared_intensity(self, simulation_params: SimulationParams, detector_image: DetectorImage, intensity_calculator: IntensityCalculator) -> SimulatedDetectorImage:
        simulated_intensity, qx_array, qy_array = intensity_calculator(simulation_params)
        return self.smear(simulated_intensity, qx_array, qy_array, detector_image)
        
    def _fit_detector_template(self, simulation_params: SimulationParams, detector_image: DetectorImage, intensity_calculator: IntensityCalculator) -> float:
        smeared_intensity = self._smeared_intensity(
            simulation_params=simulation_params,
            detector_image=detector_image,
            intensity_calculator=intensity_calculator
        )
        return self.fit_detector(detector_image, smeared_intensity)

    def fit(self, simulation_params: SimulationParams) -> float:
        return np.average(
            [self._fit_detector_template(
                simulation_params = simulation_params, 
                detector_image = simulated_detector,
                intensity_calculator=intensity_calculator) for simulated_detector, intensity_calculator in zip(self.simulated_detectors, self.intensity_calculators)]
        )

    @classmethod
    def generate_standard_fitter(cls, simulated_detectors: List[DetectorImage], box_list: List[Box], qxqy_list: List[Tuple[np.ndarray, np.ndarray]]):
        intensity_calculators = [create_intensity_calculator(box_list, qx, qy, detector_image) for (qx, qy), detector_image in zip(qxqy_list, simulated_detectors)]
        return cls(
            simulated_detectors=simulated_detectors,
            intensity_calculators=intensity_calculators,
            smear=detector_smearer,
            fit_detector=chi_squared_fit_with_default_uncertainty
        )

    @classmethod
    def generate_no_smear_fitter(cls, simulated_detectors: List[DetectorImage], box_list: List[Box]):
        intensity_calculators = [create_intensity_calculator(box_list, detector_image.qX, detector_image.qY, detector_image) for detector_image in simulated_detectors]
        return cls(
            simulated_detectors=simulated_detectors,
            intensity_calculator=intensity_calculators,
            smear=no_smearer,
            fit_detector=chi_squared_fit_with_default_uncertainty
        )



if __name__ == "__main__":
    pass