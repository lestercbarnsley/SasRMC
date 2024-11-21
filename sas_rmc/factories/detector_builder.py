#%%


from collections.abc import Callable
import pandas as pd
from pydantic.dataclasses import dataclass as pydantic_dataclass

import numpy as np

from sas_rmc import constants
from sas_rmc.detector import  DetectorImage, DetectorPixel, Polarization, subtract_detectors
from sas_rmc.factories import parse_data

PI = constants.PI

def area_to_radius(area: float) -> float: # I made this not anonymous so I can write a docstring later if needed
    return np.sqrt(area / PI)


@pydantic_dataclass
class DetectorPixelFactory: 
    qX: float
    qY: float
    intensity: float
    intensity_err: float
    qZ: float = 0
    sigma_para: float | None = None
    sigma_perp: float | None = None
    shadow_factor: bool = True

    @classmethod
    def gen_from_row(cls, row: dict):
        d = row | {
            'shadow_factor' : float(row.get('intensity', 0)) > 0
        }
        return cls(**d)
    

@pydantic_dataclass
class DetectorConfig:
    detector_distance_in_m: float
    collimation_distance_in_m: float
    collimation_aperture_area_in_m2: float
    sample_aperture_area_in_m2: float
    detector_pixel_size_in_m: float
    wavelength_in_angstrom: float
    wavelength_spread: float = 0.1
    polarization: Polarization = Polarization.UNPOLARIZED

    def get_sigma_geometric(self) -> float:
        k_mag = 2 * PI / self.wavelength_in_angstrom
        l1, l2 = self.collimation_distance_in_m, self.detector_distance_in_m
        r1, r2 = area_to_radius(self.collimation_aperture_area_in_m2), area_to_radius(self.sample_aperture_area_in_m2)
        l_dash = 1 / ((1 / l1) + (1 / l2))
        delta_d = self.detector_pixel_size_in_m
        geometric_term = 3 * ((r1 / l1)**2) + 3 * ((r2 / l_dash)**2) + ((delta_d / l2)**2)
        sigma_geom =  np.sqrt(((k_mag ** 2) / 12) * geometric_term)
        return sigma_geom

    def get_sigma_parallel(self, qx: float, qy: float) -> float:
        sigma_geom = self.get_sigma_geometric()
        q = np.sqrt(qx**2 + qy**2)
        sigma_para = np.sqrt((q * (self.wavelength_spread / 2))**2 + sigma_geom**2)
        return sigma_para
    
    def create_pixel(self, pixel_factory: DetectorPixelFactory) -> DetectorPixel: # The pixel factory coerces types as it is a Pydantic dataclass
        return DetectorPixel(
            qX=pixel_factory.qX,
            qY=pixel_factory.qY,
            intensity=pixel_factory.intensity,
            intensity_err=pixel_factory.intensity_err,
            qZ=pixel_factory.qZ,
            sigma_para=pixel_factory.sigma_para if pixel_factory.sigma_para is not None else self.get_sigma_parallel(pixel_factory.qX, pixel_factory.qY),
            sigma_perp=pixel_factory.sigma_perp if pixel_factory.sigma_perp is not None else self.get_sigma_geometric(),
            shadow_factor = pixel_factory.shadow_factor
        )
    
    @classmethod
    def gen_from_config(cls, config: dict):
        d = config | {
            'detector_distance_in_m' : config.get('Detector distance', 0),
            'collimation_distance_in_m' : config.get('Collimation distance', 0),
            'collimation_aperture_area_in_m2' : config.get('Collimation aperture', 0),
            'sample_aperture_area_in_m2' : config.get('Sample aperture', 0),
            'detector_pixel_size_in_m' : config.get('Detector pixel', 0),
            'wavelength_in_angstrom' : config.get('Wavelength', 5.0),
            'wavelength_spread' : config.get('Wavelength Spread', 0.1),
            'polarization' : config.get('Polarization', Polarization.UNPOLARIZED.value)
        }
        if d['polarization'] == 'down':
            d['polarization'] = 'spin_down'
        elif d['polarization'] == 'up':
            d['polarization'] = 'spin_up'
        return cls(**d)
    

@pydantic_dataclass
class DetectorConfigNoSmear(DetectorConfig):
    def get_sigma_geometric(self) -> float:
        return 0.0
    def get_sigma_parallel(self, qx: float, qy: float) -> float:
        return 0.0
    

def detector_image_from_dataframe(detector_df: pd.DataFrame, detector_config: DetectorConfig, polarization: Polarization | None = None):
    pixels = [detector_config.create_pixel(DetectorPixelFactory.gen_from_row({k : v for k, v in row.items()})) for _, row in detector_df.iterrows()]
    return DetectorImage(pixels, polarization=polarization if polarization is not None else detector_config.polarization)

def create_detector_image(dataframes: dict[str, pd.DataFrame], detector_config: DetectorConfig, buffer_source: str | None = None) -> DetectorImage: # raises KeyError
    detector_image = detector_image_from_dataframe(dataframes["Data Source"], detector_config)
    if not buffer_source:
        return detector_image
    if isinstance(buffer_source, float):
        return subtract_detectors(detector_image, buffer_source)
    if buffer_source in dataframes:
        buffer_image = detector_image_from_dataframe(dataframes[buffer_source], detector_config, Polarization.UNPOLARIZED)
        return subtract_detectors(detector_image, buffer_image)
    raise KeyError("Could not find buffer source")
    
def create_detector_images(dataframes: dict[str, pd.DataFrame], detector_config_creator: Callable[[dict], DetectorConfig]) -> list[DetectorImage]:
    value_frame = list(dataframes.values())[0]
    value_dict = parse_data.parse_value_frame(value_frame)
    if value_dict.get("Data Source"):
        return [create_detector_image(dataframes, detector_config_creator(value_dict), buffer_source=value_dict.get('Buffer Source'))]
    df = dataframes['Data parameters']
    return [create_detector_image(
        dataframes=dataframes,
        detector_config=detector_config_creator({k : v for k, v in row.items()}),
        buffer_source=row.get('Buffer Source')
    ) for _, row in df.iterrows()]

def create_detector_images_no_smearing(dataframes: dict[str, pd.DataFrame]) -> list[DetectorImage]:
    return create_detector_images(dataframes, DetectorConfigNoSmear.gen_from_config)

def create_detector_images_with_smearing(dataframes: dict[str, pd.DataFrame]) -> list[DetectorImage]:
    return create_detector_images(dataframes, DetectorConfig.gen_from_config)
    
    
if __name__ == "__main__":
    import inspect
    from pprint import pprint
    ### I'm keeping this here for now, because this could be a useful way to propagate
    ### Config parameters to multiple factories


    def coerce_types(func):

        def wrapper(*args, **kwargs):

            coerced_kwargs = {}
            for k, v in inspect.signature(func).parameters.items():
                if k in kwargs:
                    coerced_kwargs[k] = v.annotation(kwargs[k])
            return func(*args, **coerced_kwargs)
        return wrapper
    
    @coerce_types
    def test_1(a: float, b: float, c: float) -> float:
        return a + b + c
    
    print(test_1(**{'a' : '3.7', 'b' : '4', 'c': 7, 'd' : 8}))

    @coerce_types
    def test_with_polarization(polarization: Polarization) -> None:
        print(polarization)



# %%
