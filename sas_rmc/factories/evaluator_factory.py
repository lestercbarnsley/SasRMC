import pandas as pd
import numpy as np

from sas_rmc.detector import DetectorImage
from sas_rmc.evaluator import EvaluatorWithFitter, FitterMultiple, Smearing2DFitter, NoSmearing2DFitter, qXqY_delta
from sas_rmc.form_calculator import FieldDirection
from sas_rmc.result_calculator import AnalyticalCalculator
from sas_rmc.factories import detector_builder

def analytical_calculator_from_experimental_detector(detector: DetectorImage, density_factor: float, field_direction: FieldDirection = FieldDirection.Y) -> AnalyticalCalculator:
    qXs = np.unique(detector.qX)
    qYs = np.unique(detector.qY)
    qX_diff, qY_diff = qXqY_delta(detector)
    qX_lin = np.arange(start = qXs.min(), stop = qXs.max(), step=qX_diff / density_factor)
    qY_lin = np.arange(start = qYs.min(), stop = qYs.max(), step=qY_diff / density_factor)
    qX_arr, qY_arr = np.meshgrid(qX_lin, qY_lin)
    return AnalyticalCalculator(
        qx_array=qX_arr,
        qy_array=qY_arr,
        polarization=detector.polarization,
        field_direction=field_direction
    )

def create_evaluator_with_smearing(dataframes: dict[str, pd.DataFrame]) -> EvaluatorWithFitter:
    detector_list = detector_builder.create_detector_images(dataframes)
    return EvaluatorWithFitter(
        fitter=FitterMultiple(
            fitter_list=[Smearing2DFitter(
                result_calculator=analytical_calculator_from_experimental_detector(detector, 1.4),
                experimental_detector=detector
            ) for detector in detector_list],
            weight=[detector.shadow_factor.sum() for detector in detector_list]
        ),
    )

def create_evaluator_no_smearing(dataframes: dict[str, pd.DataFrame]) -> EvaluatorWithFitter:
    detector_list = detector_builder.create_detector_images(dataframes)
    return EvaluatorWithFitter(
        fitter=FitterMultiple(
            fitter_list=[NoSmearing2DFitter(
                result_calculator=AnalyticalCalculator(detector.qX, detector.qY, detector.polarization),
                experimental_detector=detector
            ) for detector in detector_list],
            weight=[detector.shadow_factor.sum() for detector in detector_list]
        ),
    )