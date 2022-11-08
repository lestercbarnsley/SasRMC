#%%
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Callable, Tuple, Union
from pathlib import Path

import numpy as np

from . import particle_factory, parse_data
from ..detector import DetectorImage, DetectorConfig, Polarization, SimulatedDetectorImage
from ..result_calculator import ResultCalculator, AnalyticalCalculator, NumericalCalculator
from ..vector import VectorSpace
from .. import constants

PI = constants.PI


@dataclass
class ResultMakerFactory(ABC):

    @abstractmethod
    def create_result_maker(self) -> Callable[[DetectorImage], ResultCalculator]:
        pass


def qxqy_from_detector(detector: DetectorImage, range_factor: float, resolution_factor: float, delta_qxqy_strategy: Callable[[DetectorImage], Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    detector_qx, detector_qy = detector.qX, detector.qY
    delta_qxqy_strategy = delta_qxqy_strategy if delta_qxqy_strategy is not None else (lambda d : d.qxqy_delta)
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
    particle_factory: particle_factory.ParticleFactory
    numerical_range_factor: float = 1.05
    range_factor: float = 1.2
    resolution_factor: float = 1.0

    def create_result_maker(self) -> Callable[[DetectorImage], ResultCalculator]:
        numerical_range_factor = self.numerical_range_factor
        range_factor = self.range_factor
        resolution_factor = self.resolution_factor

        def numerical_calculator_maker(detector: DetectorImage) -> NumericalCalculator:
            biggest_dimension_0, biggest_dimension_1, biggest_dimension_2 = 0, 0, 0
            for _ in range(100_000):
                particle = self.particle_factory.create_particle()
                for shape in particle.shapes:
                    d_0, d_1, d_2 = shape.dimensions
                    biggest_dimension_0 = np.max(numerical_range_factor *  d_0, biggest_dimension_0)
                    biggest_dimension_1 = np.max(numerical_range_factor *  d_1, biggest_dimension_1)
                    biggest_dimension_2 = np.max(numerical_range_factor *  d_2, biggest_dimension_2)
            vector_space_resolution = PI / np.max(detector.q)
            vector_space = VectorSpace.gen_from_bounds(
                x_min = -biggest_dimension_0 / 2, x_max = biggest_dimension_0 / 2, x_num = int(biggest_dimension_0 / vector_space_resolution),
                y_min = -biggest_dimension_1 / 2, y_max = biggest_dimension_1 / 2, y_num = int(biggest_dimension_1 / vector_space_resolution),
                z_min = -biggest_dimension_2 / 2, z_max = biggest_dimension_2 / 2, z_num = int(biggest_dimension_2 / vector_space_resolution)
                )
            qx, qy = qxqy_from_detector(detector, range_factor, resolution_factor)
            return NumericalCalculator(qx, qy, vector_space=vector_space)

        return numerical_calculator_maker


@dataclass
class DetectorImageFactory(ABC):

    @abstractmethod
    def create_detector_image(self, detector_config: DetectorConfig = None) -> DetectorImage:
        pass


@dataclass
class ConfigFactory(ABC):

    @abstractmethod
    def create_detector_config(self) -> DetectorConfig:
        pass


@dataclass
class BufferStrategy(ABC):

    @abstractmethod
    def process_detector_image(self, detector_image: DetectorImage) -> DetectorImage:
        pass


@dataclass
class DetectorBuilder:

    raw_detector_factory: DetectorImageFactory = None
    config_factory: ConfigFactory = None
    buffer_strategy: BufferStrategy = field(default_factory = lambda : BufferSubtraction(0.0, 0.0))

    def add_detector_image_factory(self, raw_detector_factory: DetectorImageFactory):
        self.raw_detector_factory = raw_detector_factory

    def add_config_factory(self, config_factory: ConfigFactory):
        self.config_factory = config_factory

    def add_buffer_strategy(self, buffer_strategy: BufferStrategy):
        self.buffer_strategy = buffer_strategy

    def build_detector_image(self) -> DetectorImage:
        detector_config = self.config_factory.create_detector_config()
        raw_detector_image = self.raw_detector_factory.create_detector_image(detector_config=detector_config)
        return self.buffer_strategy.process_detector_image(raw_detector_image)


@dataclass
class DetectorImageFromFile(DetectorImageFactory):
    file_source: Path
    simulated_detector_image: bool = True

    def create_detector_image(self, detector_config: DetectorConfig = None) -> DetectorImage:
        t = SimulatedDetectorImage if self.simulated_detector_image else DetectorImage
        return t.gen_from_txt(self.file_source, detector_config=detector_config)


@dataclass
class DetectorImageFromDataFrames(DetectorImageFactory):
    data_source: str
    dataframes: dict
    simulated_detector_image: bool = True

    def create_detector_image(self, detector_config: DetectorConfig = None) -> DetectorImage:
        data_source_df = self.dataframes[self.data_source]
        t = SimulatedDetectorImage if self.simulated_detector_image else DetectorImage
        return t.gen_from_pandas(dataframe=data_source_df, detector_config=detector_config)


POLARIZATION_DICT = {
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
        return POLARIZATION_DICT.get(polarization_str, Polarization.UNPOLARIZED)


@dataclass
class DetectorConfigFromDict(ConfigFactory):
    dataframe: dict

    def create_detector_config(self) -> DetectorConfig:
        dataframe = self.dataframe
        return DetectorConfig(
            detector_distance_in_m=dataframe.get("Detector distance", 0),
            collimation_distance_in_m=dataframe.get("Collimation distance", 0),
            collimation_aperture_area_in_m2=dataframe.get("Collimation aperture", 0),
            sample_aperture_area_in_m2=dataframe.get("Sample aperture", 0),
            detector_pixel_size_in_m=dataframe.get("Detector pixel", 0),
            wavelength_in_angstrom=dataframe.get("Wavelength", 5.0),
            wavelength_spread=dataframe.get("Wavelength Spread", 0.1),
            polarization=get_polarization(dataframe.get("Polarization", "out")),
        )


@dataclass
class BufferSubtraction(BufferStrategy):
    buffer_intensity: Union[float, np.ndarray] = 0.0
    buffer_uncertainty: Union[float, np.ndarray] = 0.0

    def process_detector_image(self, detector_image: DetectorImage) -> DetectorImage:
        intensity = detector_image.intensity - self.buffer_intensity
        intensity_uncertainty = detector_image.intensity_err - self.buffer_intensity
        detector_image.intensity = intensity
        detector_image.intensity_err = intensity_uncertainty
        return detector_image


@dataclass
class BufferImageSubtraction(BufferStrategy):
    buffer_image: DetectorImage

    def process_detector_image(self, detector_image: DetectorImage) -> DetectorImage:
        buffer_intensity = self.buffer_image.intensity if np.sum(self.buffer_image.intensity**2) else 0.0
        buffer_uncertainty = self.buffer_image.intensity_err if np.sum(self.buffer_image.intensity_err**2) else 0.0
        return BufferSubtraction(buffer_intensity=buffer_intensity, buffer_uncertainty=buffer_uncertainty).process_detector_image(detector_image)


@dataclass
class BufferStrategyFromDataFrame(BufferStrategy):
    data_source: str
    dataframes: dict

    def process_detector_image(self, detector_image: DetectorImage) -> DetectorImage:
        buffer_image = DetectorImageFromDataFrames(data_source=self.data_source, dataframes=self.dataframes).create_detector_image()
        return BufferImageSubtraction(buffer_image).process_detector_image(detector_image) 


@dataclass
class BufferStrategyFromFile(BufferStrategy):
    file_source: str

    def process_detector_image(self, detector_image: DetectorImage) -> DetectorImage:
        buffer_image = DetectorImageFromFile(file_source=self.file_source).create_detector_image()
        return BufferImageSubtraction(buffer_image).process_detector_image(detector_image)

        

@dataclass
class DetectorFromDataFrame(DetectorImageFactory):
    all_dataframes: dict
    raw_image_source: str
    buffer_source: str
    config_dataframe: dict

    def create_detector_image(self, detector_config: DetectorConfig = None) -> DetectorImage:
        builder = DetectorBuilder()
        if Path(self.raw_image_source).exists():
            builder.add_detector_image_factory(DetectorImageFromFile(self.raw_image_source))
        else:
            builder.add_detector_image_factory(DetectorImageFromDataFrames(self.raw_image_source, dataframes=self.all_dataframes))
        builder.add_config_factory(DetectorConfigFromDict(dataframe=self.config_dataframe))
        if parse_data.is_float(self.buffer_source):
            builder.add_buffer_strategy(BufferSubtraction(buffer_intensity = float(self.buffer_source)))
        elif self.buffer_source == "":
            builder.add_buffer_strategy(BufferSubtraction(buffer_intensity = 0.0))
        elif Path(self.buffer_source).exists():
            builder.add_buffer_strategy(BufferStrategyFromFile(self.buffer_source))
        else:
            builder.add_buffer_strategy(BufferStrategyFromDataFrame(self.buffer_source, dataframes = self.all_dataframes))
        return builder.build_detector_image()
       


@dataclass
class MultipleDetectorBuilder:
    dataframes: dict
    config_dict: dict

    def build_detector_images(self) -> List[DetectorImage]:
        datasources = list(self.dataframes.values())[1]
        data_source = self.config_dict.get("Data Source", "")
        if data_source:
            buffer_source = self.config_dict.get("Buffer Source", 0.0)
            return [DetectorFromDataFrame(self.dataframes, data_source, buffer_source=buffer_source, config_dataframe=self.config_dict).create_detector_image()]
       
        return [DetectorFromDataFrame(self.dataframes, row.get("Data Source", ""), row.get("Buffer Source", 0.0), config_dataframe=parse_data.dataseries_to_config_dict(row)).create_detector_image() for _, row in datasources.iterrows()]


