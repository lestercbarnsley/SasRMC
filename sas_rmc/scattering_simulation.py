#%%

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Protocol
from abc import ABC, abstractmethod

import numpy as np

from .box_simulation import Box
from .detector import DetectorImage, SimulatedDetectorImage, Polarization
from .form_calculator import box_intensity_average


@dataclass
class SimulationParams:
    rescale_factor: float = 1
    magnetic_rescale: float = 1

    def get_physical_acceptance(self):
        return self.rescale_factor > 0 and self.magnetic_rescale > 0


@dataclass
class IntensityCalculator(ABC):
    qx_array: np.ndarray
    qy_array: np.ndarray

    @abstractmethod
    def calculate_intensity(self, simulation_params: SimulationParams, detector_image: DetectorImage = None)  -> np.ndarray:
        pass


class SmearingMode(Enum):
    SMEAR = auto()
    NOSMEAR = auto()


@dataclass
class BoxIntensityCalculator(IntensityCalculator):
    box_list: List[Box]
    smearing_mode: SmearingMode = SmearingMode.SMEAR

    def _intensity_with_smear(self, rescale_factor: float = 1, magnetic_rescale: float = 1, detector_image: DetectorImage = None) -> np.ndarray:
        polarization = Polarization.UNPOLARIZED if detector_image is None else detector_image.polarization
        intensity = box_intensity_average(
            self.box_list,
            self.qx_array,
            self.qy_array,
            rescale_factor=rescale_factor,
            magnetic_rescale=magnetic_rescale,
            polarization=polarization
        )
        '''np.average(
            [box_intensity(
                box = box, 
                qx = self.qx_array, 
                qy = self.qy_array, 
                rescale_factor=rescale_factor, 
                magnetic_rescale=magnetic_rescale, 
                polarization=polarization) for box in self.box_list],
            axis=0
        )'''
        return intensity

    def _intensity_no_smear(self, rescale_factor: float = 1, magnetic_rescale: float = 1, detector_image: DetectorImage = None) -> np.ndarray:
        qx, qy, polarization = (self.qx_array, self.qy_array, Polarization.UNPOLARIZED) if detector_image is None else (detector_image.qX, detector_image.qY, detector_image.polarization)
        intensity = box_intensity_average(
            self.box_list,
            qx,
            qy,
            rescale_factor=rescale_factor,
            magnetic_rescale=magnetic_rescale,
            polarization=polarization
        )
        ''' intensity = np.average(
            [box_intensity(
                box = box,
                qx = qx, 
                qy = qy, 
                rescale_factor=rescale_factor, 
                magnetic_rescale=magnetic_rescale, 
                polarization=polarization) for box in self.box_list],
            axis=0
        )'''
        return intensity

    def calculate_intensity(self, simulation_params: SimulationParams, detector_image: DetectorImage = None) -> np.ndarray:
        intensity_method = {
            SmearingMode.SMEAR : self._intensity_with_smear,
            SmearingMode.NOSMEAR : self._intensity_no_smear
        }[self.smearing_mode]
        return intensity_method(
            rescale_factor=simulation_params.rescale_factor,
            magnetic_rescale=simulation_params.magnetic_rescale,
            detector_image=detector_image,
        )


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
    intensity_calculator: IntensityCalculator
    smear: Callable[[np.ndarray, np.ndarray, np.ndarray, DetectorImage], SimulatedDetectorImage]
    fit_detector: Callable[[DetectorImage, SimulatedDetectorImage], float]

    def _smeared_intensity(self, simulation_params: SimulationParams, detector_image: DetectorImage) -> SimulatedDetectorImage:
        simulated_intensity = self.intensity_calculator.calculate_intensity(simulation_params, detector_image)
        qx_array, qy_array = self.intensity_calculator.qx_array, self.intensity_calculator.qy_array
        return self.smear(simulated_intensity, qx_array, qy_array, detector_image)
        
    def _fit_detector_template(self, simulation_params: SimulationParams, detector_image: DetectorImage) -> float:
        smeared_intensity = self._smeared_intensity(
            simulation_params=simulation_params,
            detector_image=detector_image
        )
        return self.fit_detector(detector_image, smeared_intensity)

    def fit(self, simulation_params: SimulationParams) -> float:
        return np.average(
            [self._fit_detector_template(
                simulation_params = simulation_params, 
                detector_image = simulated_detector) for simulated_detector in self.simulated_detectors]
        )

    @classmethod
    def generate_standard_fitter(cls, simulated_detectors: List[DetectorImage], box_list: List[Box], qx_array: np.ndarray, qy_array: np.ndarray):
        return cls(
            simulated_detectors=simulated_detectors,
            intensity_calculator=BoxIntensityCalculator(
                qx_array=qx_array,
                qy_array=qy_array,
                box_list=box_list
            ),
            smear=detector_smearer,
            fit_detector=chi_squared_fit_with_default_uncertainty
        )

    @classmethod
    def generate_no_smear_fitter(cls, simulated_detectors: List[DetectorImage], box_list: List[Box]):
        qx, qy = simulated_detectors[0].qX, simulated_detectors[0].qY # these are actually never used in no smear mode
        return Fitter2D(
            simulated_detectors=simulated_detectors,
            intensity_calculator=BoxIntensityCalculator(
                qx_array=qx,
                qy_array=qy,
                box_list=box_list,
                smearing_mode=SmearingMode.NOSMEAR
            ),
            smear=no_smearer,
            fit_detector=chi_squared_fit_with_default_uncertainty
        )



if __name__ == "__main__":
    pass