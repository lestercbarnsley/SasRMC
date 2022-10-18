#%%
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Callable, Tuple, Union, Type
from pathlib import Path

import numpy as np
import pandas as pd

from .particle_factory import ParticleFactory
from ..detector import DetectorImage, DetectorConfig, Polarization, SimulatedDetectorImage
from ..result_calculator import ResultCalculator, AnalyticalCalculator, NumericalCalculator
from ..vector import VectorSpace


PI = np.pi


@dataclass
class ResultMakerFactory(ABC):

    @abstractmethod
    def create_result_maker(self) -> Callable[[DetectorImage], ResultCalculator]:
        pass


def qxqy_from_detector(detector: DetectorImage, range_factor: float, resolution_factor: float, delta_qxqy_strategy: Callable[[DetectorImage], Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    detector_qx, detector_qy = detector.qX, detector.qY
    delta_qxqy_strategy = delta_qxqy_strategy if delta_qxqy_strategy is not None else (lambda d : d.delta_qxqy_from_detector())
    detector_delta_qx, detector_delta_qy = delta_qxqy_strategy(detector)
    line_maker = lambda starting_q_min, starting_q_max, starting_q_step : np.arange(start = range_factor * starting_q_min, stop = +range_factor * starting_q_max, step = starting_q_step / resolution_factor)
    qx_linear = line_maker(np.min(detector_qx), np.max(detector_qx), detector_delta_qx)
    qy_linear = line_maker(np.min(detector_qy), np.max(detector_qy), detector_delta_qy)
    qx, qy = np.meshgrid(qx_linear, qy_linear)
    return qx, qy


@dataclass
class AnalyticalResultMakerFactory(ResultMakerFactory):
    range_factor: float = 1.2
    resolution_factor: float = 1.0

    def create_result_maker(self) -> Callable[[DetectorImage], ResultCalculator]:
        def analytical_calculator_maker(detector: DetectorImage) -> AnalyticalCalculator:
            qx, qy = qxqy_from_detector(detector, self.range_factor, self.resolution_factor)
            return AnalyticalCalculator(qx, qy)
        return analytical_calculator_maker


@dataclass
class NumericalResultMakerFactory(ResultMakerFactory):
    particle_factory: ParticleFactory
    numerical_range_factor: float = 1.05
    range_factor: float = 1.2
    resolution_factor: float = 1.0

    def create_result_maker(self) -> Callable[[DetectorImage], ResultCalculator]:

        def numerical_calculator_maker(detector: DetectorImage) -> NumericalCalculator:
            biggest_dimension_0, biggest_dimension_1, biggest_dimension_2 = 0, 0, 0
            for _ in range(100_000):
                particle = self.particle_factory.create_particle()
                for shape in particle.shapes:
                    d_0, d_1, d_2 = shape.dimensions
                    biggest_dimension_0 = np.max(self.numerical_range_factor *  d_0, biggest_dimension_0)
                    biggest_dimension_1 = np.max(self.numerical_range_factor *  d_1, biggest_dimension_1)
                    biggest_dimension_2 = np.max(self.numerical_range_factor *  d_2, biggest_dimension_2)
            vector_space_resolution = PI / np.max(detector.q)
            vector_space = VectorSpace.gen_from_bounds(
                x_min = -biggest_dimension_0 / 2, x_max = biggest_dimension_0 / 2, x_num = int(biggest_dimension_0 / vector_space_resolution),
                y_min = -biggest_dimension_1 / 2, y_max = biggest_dimension_1 / 2, y_num = int(biggest_dimension_1 / vector_space_resolution),
                z_min = -biggest_dimension_2 / 2, z_max = biggest_dimension_2 / 2, z_num = int(biggest_dimension_2 / vector_space_resolution)
                )
            qx, qy = qxqy_from_detector(detector, self.range_factor, self.resolution_factor)
            return NumericalCalculator(qx, qy, vector_space=vector_space)

        return numerical_calculator_maker


def detector_from_dataframe(data_source: str, data_frames: dict, detector_config: DetectorConfig = None) -> SimulatedDetectorImage:
    if data_source in data_frames:
        data_source_df = data_frames[data_source]
        detector = SimulatedDetectorImage.gen_from_pandas(
            dataframe=data_source_df,
            detector_config=detector_config
        )
        return detector
    if Path(data_source).exists():
        detector = SimulatedDetectorImage.gen_from_txt(
        file_location=Path(data_source),
        detector_config=detector_config
        )
        return detector
    raise Exception(f"Detector named {data_source} could not be found")

def subtract_buffer_intensity(detector: DetectorImage, buffer: Union[float, DetectorImage]) -> DetectorImage:
    if isinstance(buffer, DetectorImage):
        detector.intensity = detector.intensity - buffer.intensity
        detector.intensity_err = np.sqrt(detector.intensity_err**2 + buffer.intensity_err**2) # This is the proper way to do this, since the errors add in quadrature. If the buffer intensity error isn't present, the total error won't change
    elif isinstance(buffer, float):
        detector.intensity = detector.intensity - buffer
    detector.shadow_factor = detector.shadow_factor * (detector.intensity > 0)
    return detector

polarization_dict = {
    "down" : Polarization.SPIN_DOWN,
    "up" : Polarization.SPIN_UP,
    "unpolarized" : Polarization.UNPOLARIZED,
    "unpolarised" : Polarization.UNPOLARIZED,
    "out" : Polarization.UNPOLARIZED
}

def get_polarization(polarization_str: str) -> Polarization:
    try:
        polarization = Polarization(polarization_str)
        return polarization
    except ValueError:
        return polarization_dict.get(polarization_str, Polarization.UNPOLARIZED)


@dataclass
class DetectorDataConfig:
    data_source: str
    label: str
    detector_config: DetectorConfig = None
    buffer_source: str = ""

    def _generate_buffer(self, data_frames: dict) -> Union[float, DetectorImage]:
        if not self.buffer_source:
            return 0
        if is_float(self.buffer_source):
            return float(self.buffer_source)
        return detector_from_dataframe(
            data_source=self.buffer_source,
            detector_config=self.detector_config,
            data_frames=data_frames
        )

    def generate_detector(self, data_frames: dict) -> SimulatedDetectorImage:
        detector = detector_from_dataframe(
            data_source=self.data_source,
            detector_config=self.detector_config,
            data_frames=data_frames
        )
        buffer = self._generate_buffer(data_frames)
        return subtract_buffer_intensity(detector, buffer)

    @classmethod
    def generate_detectorconfig_from_dict(cls, config_dict: dict):
        detector_config = DetectorConfig(
            detector_distance_in_m=config_dict.get("Detector distance", 0),
            collimation_distance_in_m=config_dict.get("Collimation distance", 0),
            collimation_aperture_area_in_m2=config_dict.get("Collimation aperture", 0),
            sample_aperture_area_in_m2=config_dict.get("Sample aperture", 0),
            detector_pixel_size_in_m=config_dict.get("Detector pixel", 0),
            wavelength_in_angstrom=config_dict.get("Wavelength", 5.0),
            wavelength_spread=config_dict.get("Wavelength Spread", 0.1),
            polarization=get_polarization(config_dict.get("Polarization", "out")),
        )
        return cls(
            data_source=config_dict.get("Data Source", ""),
            label = config_dict.get("Label", ""),
            detector_config=detector_config,
            buffer_source=config_dict.get("Buffer Source", "")
        )

    @classmethod
    def generate_detectorconfig_list_from_dataframe(cls, dataframe_2: pd.DataFrame) -> List:
        return [cls.generate_detectorconfig_from_dict(series_to_config_dict(row)) for _, row in dataframe_2.iterrows()]




