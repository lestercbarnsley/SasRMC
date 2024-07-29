#%%


import pandas as pd
from pydantic.dataclasses import dataclass

from sas_rmc import constants
from sas_rmc.detector import DetectorConfig, DetectorImage, DetectorPixel, Polarization, subtract_two_detectors
from sas_rmc.factories import parse_data

PI = constants.PI




@dataclass
class DetectorPixelFactory:
    qX: float
    qY: float
    intensity: float
    intensity_err: float
    qZ: float = 0
    shadow_factor: bool = True

    def create_pixel(self, detector_config: DetectorConfig) -> DetectorPixel:
        return DetectorPixel(
            qX=self.qX,
            qY=self.qY,
            intensity=self.intensity,
            intensity_err=self.intensity_err,
            qZ=self.qZ,
            sigma_para=detector_config.get_sigma_parallel(qx=self.qX, qy= self.qY),
            sigma_perp=detector_config.get_sigma_geometric(),
            shadow_factor = self.shadow_factor
        )
    
    @classmethod
    def gen_from_row(cls, row: dict):
        d = row | {
            'shadow_factor' : float(row.get('intensity', 0)) > 0
        }
        return cls(**d)
        

@dataclass
class DetectorConfigFactory:
    detector_distance_in_m: float
    collimation_distance_in_m: float
    collimation_aperture_area_in_m2: float
    sample_aperture_area_in_m2: float
    detector_pixel_size_in_m: float
    wavelength_in_angstrom: float
    wavelength_spread: float = 0.1
    polarization: str = Polarization.UNPOLARIZED.value

    def create_detector_config(self) -> DetectorConfig:
        polarization = Polarization(self.polarization)
        d = self.__dict__ | {'polarization' : polarization}
        return DetectorConfig(**d)
    
    @classmethod
    def gen_from_config(cls, config: dict):
        d = config | {
            'detector_distance_in_m' : config['Detector distance'],
            'collimation_distance_in_m' : config['Collimation distance'],
            'collimation_aperture_area_in_m2' : config['Collimation aperture'],
            'sample_aperture_area_in_m2' : config['Sample aperture'],
            'detector_pixel_size_in_m' : config['Detector pixel'],
            'wavelength_in_angstrom' : config['Wavelength'],
            'wavelength_spread' : config['Wavelength Spread'],
            'polarization' : config['Polarization']
        }
        if d['polarization'] == 'down':
            d['polarization'] = 'spin_down'
        elif d['polarization'] == 'up':
            d['polarization'] = 'spin_up'
        return cls(**d)
    

def create_detector_image(dataframe: pd.DataFrame, value_dict: dict) -> DetectorImage:
    detector_config = DetectorConfigFactory.gen_from_config(value_dict).create_detector_config()
    pixels = [DetectorPixelFactory.gen_from_row({k : v for k, v in row.items()}).create_pixel(detector_config) for _, row in dataframe.iterrows()]
    return DetectorImage(pixels, polarization=detector_config.polarization)


def create_detector_images(dataframes: dict[str, pd.DataFrame]) -> list[DetectorImage]:
    value_frame = list(dataframes.values())[0]
    value_dict = {k : v for k, v in parse_data.parse_value_frame(value_frame)}
    if value_dict.get("Data Source"):
        data_dict = dataframes[value_dict.get("Data Source")]
        detector = create_detector_image(data_dict, value_dict)
        buffer_source = value_dict.get("Buffer Source")
        if buffer_source:
            buffer_dict = dataframes[value_dict.get("Buffer Source")]
            buffer = create_detector_image(buffer_dict, value_dict)
            return [subtract_two_detectors(detector, buffer)]
        return [detector]
    
    
if __name__ == "__main__":
    pass


# %%
