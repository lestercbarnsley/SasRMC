import pandas as pd
import numpy as np

from sas_rmc.detector import DetectorImage
from sas_rmc.evaluator import EvaluatorWithFitter, Smearing2dFitterMultiple, Smearing2DFitter
from sas_rmc.form_calculator import FieldDirection
from sas_rmc.result_calculator import AnalyticalCalculator

def analytical_calculator_from_experimental_detector(detector: DetectorImage, density_factor: float, field_direction: FieldDirection = FieldDirection.Y) -> AnalyticalCalculator:
    qXs = np.unique(detector.qX)
    qYs = np.unique(detector.qY)
    qX_diff = np.max(np.gradient(qXs))
    qY_diff = np.max(np.gradient(qYs))
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
    
    return EvaluatorWithFitter(
        fitter=Smearing2dFitterMultiple(
            fitter_list=Smearing2DFitter(
                result_calculator=AnalyticalCalculator()
            )
        )
    )