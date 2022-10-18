#%%
from typing import Callable, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

from .box_simulation import Box
from .form_calculator import box_intensity_average
from .scattering_simulation import SimulationParams, MAGNETIC_RESCALE, NUCLEAR_RESCALE
from .result_calculator import ResultCalculator
from .detector import DetectorImage, SimulatedDetectorImage, Polarization

IntensityCalculator = Callable[[SimulationParams], np.ndarray]
ArraysFitter = Callable[[List[np.ndarray]], float]

def intensity_calculator(box_list: List[Box], result_calculator: ResultCalculator, simulation_params: SimulationParams, polarization: Polarization) -> np.ndarray:
    rescale_factor = simulation_params.get_value(NUCLEAR_RESCALE, default=1.0)
    magnetic_rescale = simulation_params.get_value(MAGNETIC_RESCALE, default=1.0)
    intensity = box_intensity_average(box_list, result_calculator, rescale_factor=rescale_factor, magnetic_rescale=magnetic_rescale, polarization=polarization)
    return intensity

def create_intensity_calculator(box_list: List[Box], result_calculator: ResultCalculator, polarization: Polarization) -> IntensityCalculator:
    return lambda simulation_params : intensity_calculator(box_list, result_calculator, simulation_params, polarization)

def default_detector_to_weighting_function(detector: DetectorImage) -> np.ndarray:
    intensity_err = detector.intensity_err
    if np.sum(intensity_err**2) != 0:
        return intensity_err
    else:
        rectified_intensity = np.where(detector.intensity > 0, detector.intensity, np.inf)
        return np.sqrt(rectified_intensity)

def smear_simulated_intensity(unsmeared_intensity: np.ndarray, simulated_qx: np.ndarray, simulated_qy: np.ndarray, simulated_detector: SimulatedDetectorImage) -> np.ndarray:
    return simulated_detector.smear(
        intensity = unsmeared_intensity,
        qx_array = simulated_qx,
        qy_array = simulated_qy
    )

def set_unsmeared_intensity(unsmeared_intensity: np.ndarray, simulated_qx: np.ndarray, simulated_qy: np.ndarray, simulated_detector: SimulatedDetectorImage) -> np.ndarray:
    simulated_detector.simulated_intensity = unsmeared_intensity
    return unsmeared_intensity

def intensity_calculator_and_smearer_from_detector(detector: SimulatedDetectorImage, box_list: List[Box], result_calculator_maker: Callable[[DetectorImage], ResultCalculator]):
    result_calculator = result_calculator_maker(detector)
    intensity_calculator_fn = create_intensity_calculator(box_list, result_calculator, detector.polarization)
    smearer = lambda simulated_intensity: smear_simulated_intensity(simulated_intensity, result_calculator.qx_array, result_calculator.qy_array, detector)
    return lambda simulation_params : smearer(intensity_calculator_fn(simulation_params))

def intensity_calculator_no_smearer(detector: SimulatedDetectorImage, box_list: List[Box], result_calculator_maker: Callable[[DetectorImage], ResultCalculator]):
    result_calculator = result_calculator_maker(detector)
    qX, qY, shadow_factor = detector.qX, detector.qY, detector.shadow_factor
    result_calculator.qx_array = qX
    result_calculator.qy_array = qY
    intensity_calculator_fn = create_intensity_calculator(box_list, result_calculator, detector.polarization)
    intensity_setter = lambda simulated_intensity: set_unsmeared_intensity(simulated_intensity * shadow_factor, qX, qY, detector)
    return lambda simulated_params : intensity_setter(intensity_calculator_fn(simulated_params))

def total_chi_squared(experimental_intensity: np.ndarray, simulated_intensity: np.ndarray, experimental_uncertainty: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    reduced_array = np.where(mask, (experimental_intensity - simulated_intensity)**2 / experimental_uncertainty**2, 0)
    return np.sum(reduced_array), np.sum(np.where(mask, 1, 0))

def average_chi_squared_fitter(experimental_detectors: List[DetectorImage], simulated_intensity_list: List[np.ndarray], weighting_function: Callable[[DetectorImage], np.ndarray], masking_function: Optional[Callable[[DetectorImage], np.ndarray]] = None) -> float:
    if masking_function is None:
        masking_function = lambda detector : detector.shadow_factor
    total_chi_squared_tuples = [total_chi_squared(detector.intensity, simulated_intensity, weighting_function(detector), masking_function(detector) ) for detector, simulated_intensity in zip(experimental_detectors, simulated_intensity_list)]
    chi_squared_sum = np.sum([chi for chi, n in total_chi_squared_tuples])
    n_sum = np.sum([n for chi, n in total_chi_squared_tuples]) # make the list once but call it twice, so I don't think lazy iteration is efficient here
    return chi_squared_sum / n_sum


@dataclass
class Fitter2D:
   
    intensity_calculators: List[IntensityCalculator] # I could make this a composite function but I want to make it clear that smearing is happening
    arrays_fitter: ArraysFitter

    def fit(self, simulation_params: SimulationParams) -> float:
        return self.arrays_fitter(
            [intensity_calcator(simulation_params) for intensity_calcator in self.intensity_calculators]
            )
    
    @classmethod
    def generate_standard_fitter(cls, detector_list: List[SimulatedDetectorImage], box_list: List[Box], result_calculator_maker: Callable[[DetectorImage], ResultCalculator], weighting_function: Callable[[DetectorImage], np.ndarray] = None, masking_function: Callable[[DetectorImage], np.ndarray] = None, smearing: bool = True):
        if weighting_function is None:
            weighting_function = default_detector_to_weighting_function
        if masking_function is None:
            masking_function = lambda detector : detector.shadow_factor
        calculating_function = intensity_calculator_and_smearer_from_detector if smearing else intensity_calculator_no_smearer
        intensity_calculators = [calculating_function(detector, box_list, result_calculator_maker) for detector in detector_list]
        arrays_fitter = lambda simulated_intensities : average_chi_squared_fitter(detector_list, simulated_intensities, weighting_function, masking_function)
        return cls(
            intensity_calculators=intensity_calculators,
            arrays_fitter=arrays_fitter
        )

