import pandas as pd
import numpy as np

from sas_rmc import polarizer
from sas_rmc.constants import np_sum
from sas_rmc.detector import DetectorImage, Polarization
from sas_rmc.evaluator import EvaluatorWithFitter, FitterMultiple, Smearing2DFitter, NoSmearing2DFitter, qXqY_delta
from sas_rmc.result_calculator import AnalyticalCalculator
from sas_rmc.factories import detector_builder


def analytical_calculator_from_experimental_detector(detector: DetectorImage, density_factor: float, polarizer: polarizer.Polarizer) -> AnalyticalCalculator:
    qXs = np.unique(detector.qX)
    qYs = np.unique(detector.qY)
    qX_diff, qY_diff = qXqY_delta(detector)
    qX_lin = np.arange(start = qXs.min(), stop = qXs.max(), step=qX_diff / density_factor)
    qY_lin = np.arange(start = qYs.min(), stop = qYs.max(), step=qY_diff / density_factor)
    qX_arr, qY_arr = np.meshgrid(qX_lin, qY_lin)
    return AnalyticalCalculator(
        qx_array=qX_arr,
        qy_array=qY_arr,
        polarizer=polarizer
    )

def polarizer_from(polarization: Polarization, field_direction_str: str) -> polarizer.Polarizer:
    field_direction = {
        'X' : polarizer.FieldDirectionX(),
        'Y' : polarizer.FieldDirectionY(),
        'Z' : polarizer.FieldDirectionZ()
    }[field_direction_str]
    return {
        Polarization.MINUS_MINUS : polarizer.PolarizerMinusMinus(field_direction),
        Polarization.MINUS_PLUS : polarizer.PolarizerMinusPlus(field_direction),
        Polarization.PLUS_MINUS : polarizer.PolarizerPlusMinus(field_direction),
        Polarization.PLUS_PLUS : polarizer.PolarizerPlusPlus(field_direction),
        Polarization.SPIN_DOWN : polarizer.PolarizerSpinDown(field_direction),
        Polarization.SPIN_UP : polarizer.PolarizerSpinUp(field_direction),
        Polarization.UNPOLARIZED : polarizer.PolarizerUnpolarized(field_direction)
    }[polarization]

def create_smearing_fitter_from_experimental_detector(detector: DetectorImage, density_factor: float = 1.4, field_direction_str: str = "Y") -> Smearing2DFitter:
    polarizer = polarizer_from(detector.polarization, field_direction_str)
    analytical_calculator = analytical_calculator_from_experimental_detector(detector, density_factor, polarizer)
    qx_matrix = analytical_calculator.qx_array
    qy_matrix = analytical_calculator.qy_array
    return Smearing2DFitter(
        result_calculator=analytical_calculator,
        experimental_detector=detector,
        qx_matrix=qx_matrix,
        qy_matrix=qy_matrix
    )

def create_evaluator_with_smearing(dataframes: dict[str, pd.DataFrame]) -> EvaluatorWithFitter:
    detector_list = detector_builder.create_detector_images_with_smearing(dataframes)
    density_factor = 1.4
    field_direction_str ="Y"
    return EvaluatorWithFitter(
        fitter=FitterMultiple(
            fitter_list=[
                create_smearing_fitter_from_experimental_detector(detector, density_factor, field_direction_str) 
                for detector in detector_list
            ],
            weight=[np_sum(detector.shadow_factor) for detector in detector_list]
        ),
    )

def create_evaluator_no_smearing(dataframes: dict[str, pd.DataFrame]) -> EvaluatorWithFitter:
    detector_list = detector_builder.create_detector_images_no_smearing(dataframes)
    field_direction_str ="Y"
    return EvaluatorWithFitter(
        fitter=FitterMultiple(
            fitter_list=[NoSmearing2DFitter(
                result_calculator=AnalyticalCalculator(
                    qx_array=detector.qX, 
                    qy_array=detector.qY, 
                    polarizer=polarizer_from(detector.polarization, field_direction_str)
                    ),
                experimental_detector=detector
            ) for detector in detector_list],
            weight=[np_sum(detector.shadow_factor) for detector in detector_list]
        ),
    )