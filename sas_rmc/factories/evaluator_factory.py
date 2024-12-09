#%%
from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd
import numpy as np
from pydantic.dataclasses import dataclass as pydantic_dataclass

from sas_rmc import polarizer
from sas_rmc.constants import np_sum, PI
from sas_rmc.detector import DetectorImage, Polarization
from sas_rmc.evaluator import Evaluator, EvaluatorWithFitter, FitterMultiple, Smearing2DFitter, NoSmearing2DFitter, qXqY_delta
from sas_rmc.result_calculator import AnalyticalCalculator
from sas_rmc.factories import detector_builder, parse_data


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

def field_direction_from(field_direction_str: str) -> polarizer.FieldDirection:
    match field_direction_str:
        case 'X':
            return polarizer.FieldDirectionX()
        case 'Y':
            return polarizer.FieldDirectionY()
        case 'Z':
            return polarizer.FieldDirectionZ()
        case _:
            raise ValueError("This field direction string is not recognized")

def polarizer_from(polarization: Polarization, field_direction_str: str) -> polarizer.Polarizer:
    field_direction = field_direction_from(field_direction_str)
    match polarization:
        case Polarization.MINUS_MINUS:
            return polarizer.PolarizerMinusMinus(field_direction)
        case Polarization.MINUS_PLUS:
            return polarizer.PolarizerMinusPlus(field_direction)
        case Polarization.PLUS_PLUS:
            return polarizer.PolarizerPlusPlus(field_direction)
        case Polarization.PLUS_MINUS:
            return polarizer.PolarizerPlusMinus(field_direction)
        case Polarization.SPIN_DOWN:
            return polarizer.PolarizerSpinDown(field_direction)
        case Polarization.SPIN_UP:
            return polarizer.PolarizerSpinUp(field_direction)
        case Polarization.UNPOLARIZED:
            return polarizer.PolarizerUnpolarized(field_direction)

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


@pydantic_dataclass
class EvaluatorFactory(ABC):
    @abstractmethod
    def create_evaluator(self) -> Evaluator:
        pass

    @abstractmethod
    def default_box_dimensions(self) -> list[float]:
        pass


@pydantic_dataclass
class EvaluatorWithSmearingFactory(EvaluatorFactory):
    detector_list: list[DetectorImage]
    detector_smearing: bool
    field_direction: str = "Y"
    density_factor: float = 1.4

    def create_smearing_evaluator(self) -> EvaluatorWithFitter:
        return EvaluatorWithFitter(
        fitter=FitterMultiple(
            fitter_list=[
                create_smearing_fitter_from_experimental_detector(detector, self.density_factor, self.field_direction) 
                for detector in self.detector_list
            ],
            weight=[np_sum(detector.shadow_factor) for detector in self.detector_list]
            ),
        )
    
    def create_nonsmearing_evaluator(self) -> EvaluatorWithFitter:
        return EvaluatorWithFitter(
            fitter=FitterMultiple(
                fitter_list=[NoSmearing2DFitter(
                    result_calculator=AnalyticalCalculator(
                        qx_array=detector.qX, 
                        qy_array=detector.qY, 
                        polarizer=polarizer_from(detector.polarization, self.field_direction)
                        ),
                    experimental_detector=detector
                ) for detector in self.detector_list],
                weight=[np_sum(detector.shadow_factor) for detector in self.detector_list]
            ),
        )

    def create_evaluator(self) -> EvaluatorWithFitter:
        if self.detector_smearing:
            return self.create_smearing_evaluator()
        return self.create_nonsmearing_evaluator() #It's ok to have conditionals in a factory
    
    def default_box_dimensions(self) -> list[float]:
        qXqY_delta_list = [qXqY_delta(experimental_detector) for experimental_detector in self.detector_list]
        smallest_qxqy = min(qXqY_delta_list, key = lambda delta : delta[0] + delta[1])
        qX_diff, qY_diff = smallest_qxqy
        return [2 * PI / qX_diff, 2 * PI / qY_diff, 2 * PI / qX_diff]
        
    @classmethod
    def create_from_dataframes(cls, dataframes: dict[str, pd.DataFrame]):
        detector_list = detector_builder.create_detector_images_with_smearing(dataframes)
        value_frame = parse_data.parse_value_frame(dataframes['Simulation parameters'])
        return cls(detector_list=detector_list, **value_frame)
    

class ProfileType(Enum):
    DETECTOR_IMAGE = "detector_image"
    PROFILE = "profile"


def infer_profile_type_from_dataframe(dataframe: pd.DataFrame) -> ProfileType:
    if all(col_name in [col.lower() for col in dataframe.columns] for col_name in ['qx', 'qy']):
        return ProfileType.DETECTOR_IMAGE
    elif 'q' in [col.lower() for col in dataframe.columns]:
        return ProfileType.PROFILE
    raise ValueError("Cannot infer profile type from headers in detector data sheets")

def infer_profile_type(dataframes: dict[str, pd.DataFrame]) -> ProfileType:
    value_frame = parse_data.parse_value_frame(dataframes['Simulation parameters'])
    data_source = value_frame.get("Data Source")
    if data_source:
        return infer_profile_type_from_dataframe(dataframes[data_source])
    data_params = dataframes['Data parameters']
    data_source = data_params['Data Source'].iloc[0]
    return infer_profile_type_from_dataframe(dataframes[data_source])

def create_evaluator_factory_from(dataframes: dict[str, pd.DataFrame]) -> EvaluatorFactory:
    profile_type = infer_profile_type(dataframes)
    match profile_type:
        case ProfileType.DETECTOR_IMAGE:
            return EvaluatorWithSmearingFactory.create_from_dataframes(dataframes)
        case ProfileType.PROFILE:
            raise NotImplementedError("To be done")
    
    
#%%
if __name__ == "__main__":
    from pathlib import Path
    input_config_path = Path(r"E:\Programming\SasRMC\data\CoreShell Simulation Input - Copy - Copy.xlsx")
    dataframes = pd.read_excel(
        input_config_path,
        dtype = str,
        sheet_name = None,
        keep_default_na=False,
        )
    
    t = EvaluatorWithSmearingFactory.create_from_dataframes(dataframes)

