#%%
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from sas_rmc.array_cache import array_cache, method_array_cache
from sas_rmc.vector import Vector
from sas_rmc import constants


PI = constants.PI
DEFAULT_SLICING_FRACTION_DENOM = 4 # This considers 6.25% of the detector (i.e, 1/4^2) per pixel, which is more than sufficient

# name string constants, this is the source of truth for the names of these quantities
QX = 'qX'#: self.qX,
QY = 'qY'#: self.qY,
INTENSITY = 'intensity'#: self.intensity,
INTENSITY_ERROR = 'intensity_err'#: self.intensity_err,
QZ = 'qZ'#: self.qZ,
SIGMA_PARA = 'sigma_para'#: self.sigma_para,
SIGMA_PERP = 'sigma_perp'#: self.sigma_perp,
SHADOW_FACTOR = 'shadow_factor'#: int(self.shadow_factor),
SIMULATED_INTENSITY = 'simulated_intensity'#: self.simulated_intensity,
SIMUATED_INTENSITY_ERR = 'simulated_intensity_err'#: self.simulated_intensity_err,
POLARIZATION = "Polarization"

def get_slicing_func_from_gaussian(gaussian: np.ndarray, slicing_range: int | None = None) -> Callable[[np.ndarray], np.ndarray]:
    i_of_max, j_of_max = np.where(gaussian == np.amax(gaussian))
    i_of_max, j_of_max = i_of_max[0], j_of_max[0]
    slicing_range = slicing_range if slicing_range is not None else int(np.max(gaussian.shape) / DEFAULT_SLICING_FRACTION_DENOM)
    i_min = max(i_of_max - int(slicing_range / 2), 0)
    i_max = min(i_min + slicing_range, gaussian.shape[0] - 1)
    i_min = i_max - slicing_range
    j_min = max(j_of_max - int(slicing_range / 2), 0)
    j_max = min(j_min + slicing_range, gaussian.shape[1] - 1)
    j_min = j_max - slicing_range
    def slicing_func(arr: np.ndarray) -> np.ndarray:
        return arr[i_min:i_max, j_min:j_max]
    return slicing_func

def test_uniques(test_space: np.ndarray, arr: np.ndarray) -> np.ndarray:
    closest_in_space = lambda v : test_space[np.argmin(np.abs(test_space - v))]
    return np.array([closest_in_space(a) for a in arr]) # I can't use broadcast because the pandas method passes in a data series rather than a numpy array
    
def fuzzy_unique(arr: np.ndarray, array_filterer: Callable[[np.ndarray], np.ndarray] = None) -> np.ndarray:
    unique_arr = np.unique(arr)
    test_range = 4 * int(np.sqrt(arr.shape[0]))
    if unique_arr.shape[0] < test_range:
        return unique_arr

    test_space_maker = lambda num : np.linspace(np.min(arr), np.max(arr), num = num) if num !=0 else np.array([np.inf])
    filtered_array = array_filterer(arr) if array_filterer else arr
    for i in range(5, test_range):#_ in enumerate(first_guess):
        test_space = test_space_maker(i)
        closest_arr = test_uniques(test_space, filtered_array)
        if not all(t in closest_arr for t in test_space):
            return test_space_maker(i-1)
    raise Exception("Unable to find enough unique values in Q-space to map detector to 2-D grid")


def average_uniques(linear_arr: np.ndarray) -> np.ndarray:
    sorted_uniques = np.sort(np.unique(linear_arr))
    if len(sorted_uniques) < 5 * np.sqrt(len(linear_arr)):
        return sorted_uniques
    diffs = np.diff(sorted_uniques)
    in_range = lambda x : -np.std(diffs) < (x - np.average(diffs)) < +np.std(diffs)
    fuzzy_uniques = []
    fuzzy_row = []
    for x, dx in zip(sorted_uniques, np.append(diffs, np.inf)):
        fuzzy_row.append(x)
        if not in_range(dx):
            fuzzy_uniques.append(np.average(fuzzy_row))
            fuzzy_row.clear()
    return np.array(fuzzy_uniques)

    
def area_to_radius(area: float): # I made this not anonymous so I can write a docstring later if needed
    return np.sqrt(area / PI)

class Polarization(Enum):
    UNPOLARIZED = "unpolarized"
    SPIN_UP = "spin_up"
    SPIN_DOWN = "spin_down"
    MINUS_MINUS = "minus_minus"
    MINUS_PLUS = "minus_plus"
    PLUS_MINUS = "plus_minus"
    PLUS_PLUS = "plus_plus"


@dataclass
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
    
    @classmethod
    def gen_from_dict(cls, d: dict):
        polarization = Polarization(d.get(POLARIZATION))
        data = d | {POLARIZATION : polarization}
        return constants.validate_fields(cls, data)



def orthogonal_xy(x: float, y: float) -> tuple[float, float]: # If I find I need this function a lot, I'll make it a method in the Vector class, but for now I'm happy for it to be a helper function
    if not np.sqrt(x**2 + y**2):
        return 0, 0
    orth_y = 0.0 if x == 0 else np.sqrt(1 / (1 + (y / x)**2)) * np.sign(x)
    orth_x = -1.0 if x == 0 else -(y / x) * orth_y
    return orth_x, orth_y


@dataclass
class DetectorPixel:
    qX: float
    qY: float
    intensity: float
    intensity_err: float
    qZ: float = 0
    sigma_para: float = 0
    sigma_perp: float = 0
    shadow_factor: bool = True

    @property
    def q_vector(self) -> Vector:
        return Vector(self.qX, self.qY, self.qZ)
    
    #@method_array_cache
    def resolution_function(self, qx_array: np.ndarray, qy_array: np.ndarray) -> np.ndarray:
        qx_offset = qx_array - self.qX
        qy_offset = qy_array - self.qY
        q_offset = (qx_offset, qy_offset, 0)
        q_vec = Vector(self.qX, self.qY)
        q_para = q_vec.unit_vector
        q_perp = Vector(*orthogonal_xy(q_para.x, q_para.y))
        q_para_arr = q_para.unit_vector * q_offset
        q_perp_arr = q_perp.unit_vector * q_offset
        gaussian = np.exp(-(1/2) * ((q_para_arr / self.sigma_para)**2 + (q_perp_arr / self.sigma_perp)**2) )

        return gaussian / np.sum(gaussian)
    
    #@method_array_cache
    def get_slicing_func(self, qx_array: np.ndarray, qy_array: np.ndarray, slicing_range: int | None = None) -> Callable[[np.ndarray], np.ndarray]:
        gaussian = self.resolution_function(qx_array, qy_array)
        return get_slicing_func_from_gaussian(gaussian, slicing_range)

    def to_dict(self):
        return {
            QX : self.qX,
            QY : self.qY,
            INTENSITY: self.intensity,
            INTENSITY_ERROR: self.intensity_err,
            QZ : self.qZ,
            SIGMA_PARA : self.sigma_para,
            SIGMA_PERP: self.sigma_perp,
            SHADOW_FACTOR : int(self.shadow_factor)
        }

    @classmethod
    def row_to_pixel(cls, data_row: dict, detector_config: DetectorConfig = None):
        qx_i = data_row[QX]
        qy_i = data_row[QY]
        intensity = data_row[INTENSITY]
        intensity_error = data_row.get(INTENSITY_ERROR, 0)
        qZ = data_row.get(QZ, 0)
        sigma_para = data_row.get(SIGMA_PARA, 0) if detector_config is None else detector_config.get_sigma_parallel(qx_i, qy_i)
        sigma_perp = data_row.get(SIGMA_PERP, 0) if detector_config is None else detector_config.get_sigma_geometric()
        shadow_factor = bool(data_row.get(SHADOW_FACTOR,bool(intensity)))
        return constants.validate_fields(cls,  {
            QX : qx_i,
            QY : qy_i,
            INTENSITY : intensity,
            INTENSITY_ERROR : intensity_error,
            QZ : qZ,
            SIGMA_PARA : sigma_para,
            SIGMA_PERP : sigma_perp,
            SHADOW_FACTOR : shadow_factor
            })


def get_pixel_qX(pixel: DetectorPixel) -> float:
    return pixel.qX

def get_pixel_qY(pixel: DetectorPixel) -> float:
    return pixel.qY

def get_pixel_intensity(pixel: DetectorPixel) -> float:
    return pixel.intensity

def get_pixel_intensity_err(pixel: DetectorPixel) -> float:
    return pixel.intensity_err

def get_pixel_qZ(pixel: DetectorPixel) -> float:
    return pixel.qZ

def get_pixel_sigma_para(pixel: DetectorPixel) -> float:
    return pixel.sigma_para

def get_pixel_sigma_perp(pixel: DetectorPixel) -> float:
    return pixel.sigma_perp

def get_pixel_shadow_factor(pixel: DetectorPixel) -> bool:
    return pixel.shadow_factor

def get_nearest_pixel(qx: float, qy: float, pixels: Iterable[DetectorPixel]) -> DetectorPixel:
    return min(pixels, key = lambda p : (qx - p.qX)**2 + (qy - p.qY)**2)

def subtract_nearest_pixel(pixel: DetectorPixel, pixels: Iterable[DetectorPixel]) -> DetectorPixel:
    qX = pixel.qX
    qY = pixel.qY
    nearest_pixel = get_nearest_pixel(qX, qY, pixels)
    return DetectorPixel(
        qX = qX,
        qY = qY,
        intensity = pixel.intensity - nearest_pixel.intensity,
        intensity_err = np.sqrt(pixel.intensity_err**2 + nearest_pixel.intensity_err**2),
        qZ = pixel.qZ,
        sigma_para=pixel.sigma_para,
        sigma_perp=pixel.sigma_perp,
        shadow_factor=pixel.shadow_factor and nearest_pixel.shadow_factor
    )

def subtract_background(pixel: DetectorPixel, background: float) -> DetectorPixel:
    return DetectorPixel(
        qX = pixel.qX,
        qY = pixel.qY,
        intensity = pixel.intensity - background,
        intensity_err = pixel.intensity_err,
        qZ = pixel.qZ,
        sigma_para=pixel.sigma_para,
        sigma_perp=pixel.sigma_perp,
        shadow_factor=pixel.shadow_factor
    )


@dataclass
class DetectorImage: # Major refactor needed for detector image, as it shouldn't be responsible for how it's plotted
    detector_pixels: list[DetectorPixel]
    polarization: Polarization = Polarization.UNPOLARIZED

    @method_array_cache
    def array_from_pixels(self, pixel_func: Callable[[DetectorPixel], Any]) -> np.ndarray:
        return np.array([pixel_func(pixel) for pixel in self.detector_pixels])

    @property
    def qX(self) -> np.ndarray:
        return self.array_from_pixels(get_pixel_qX)

    @property
    def qY(self) -> np.ndarray:
        return self.array_from_pixels(get_pixel_qY)

    @property
    def q(self) -> np.ndarray:
        return np.sqrt(self.qX**2 + self.qY**2)

    @property
    def intensity(self) -> np.ndarray:
        return self.array_from_pixels(get_pixel_intensity)

    @property
    def intensity_err(self) -> np.ndarray:
        return self.array_from_pixels(get_pixel_intensity_err)

    @property
    def qZ(self) -> np.ndarray:
        return self.array_from_pixels(get_pixel_qZ)

    @property
    def sigma_para(self) -> np.ndarray:
        return self.array_from_pixels(get_pixel_sigma_para)

    @property
    def sigma_perp(self) -> np.ndarray:
        return self.array_from_pixels(get_pixel_sigma_perp)

    @property
    def shadow_factor(self) -> np.ndarray:
        return self.array_from_pixels(get_pixel_shadow_factor)
    
    def get_loggable_data(self) -> list[dict]:
        return [pixel.to_dict() for pixel in self.detector_pixels]
    
    

    @classmethod
    def gen_from_data(cls, data_dict: dict, detector_config: DetectorConfig = None):
        qX_data = data_dict[QX]
        qY_data = data_dict[QY]
        qX_1d = fuzzy_unique(qX_data, lambda a: a[np.abs(qY_data) < 0.025 * np.max(qY_data)])#.tolist() Ultimately, this should be a strategy somehow
        qY_1d = fuzzy_unique(qY_data, lambda a: a[np.abs(qX_data) < 0.025 * np.max(qX_data)])#.tolist()
        qx, qy = np.meshgrid(qX_1d, qY_1d)
        blank_canvas_pixel = lambda qx_i, qy_i : DetectorPixel.row_to_pixel({QX : qx_i, QY : qy_i, INTENSITY : 0}, detector_config= detector_config)
        make_blank_canvas = np.frompyfunc(blank_canvas_pixel, 2, 1)
        detector_pixels = make_blank_canvas(qx, qy).astype(object)
        for index, _ in enumerate(qX_data):
            get_item = lambda k, v = None: data_dict.get(k, v)[index]
            qX, qY = get_item(QX), get_item(QY)
            row = {
                QX : qX,
                QY : qY,
                INTENSITY : get_item(INTENSITY),
                INTENSITY_ERROR : get_item(INTENSITY_ERROR),
                QZ : get_item(QZ),
                SIGMA_PARA: get_item(SIGMA_PARA),
                SIGMA_PERP: get_item(SIGMA_PERP),
                SHADOW_FACTOR : get_item(SHADOW_FACTOR),
                SIMULATED_INTENSITY : get_item(SIMULATED_INTENSITY)
            }
            i = np.argmin(np.abs(qX_1d - qX)) # This now finds the closest value rather than an exact match, then it overwrites the pixel
            j = np.argmin(np.abs(qY_1d - qY))
            detector_pixels[j, i] = DetectorPixel.row_to_pixel(row, detector_config=detector_config)
        polarization = detector_config.polarization if detector_config else Polarization(data_dict.get(POLARIZATION, Polarization.UNPOLARIZED.value))
        return cls(
            _detector_pixels = detector_pixels,
            polarization = polarization
        )

    @classmethod
    def gen_from_data(cls, data_dict: dict, detector_config: DetectorConfig = None):
        qX_data = data_dict[QX]
        get_row = lambda i : {k : data_dict.get(k)[i] for k in [QX, QY, INTENSITY, INTENSITY_ERROR, QZ, SIGMA_PARA, SIGMA_PERP, SHADOW_FACTOR]}
        detector_pixels = np.array([DetectorPixel.row_to_pixel(get_row(i), detector_config=detector_config) for i, _ in enumerate(qX_data)])
        polarization = detector_config.polarization if detector_config else Polarization(data_dict.get(POLARIZATION, Polarization.UNPOLARIZED.value))
        return cls(
            _detector_pixels = detector_pixels,
            polarization = polarization
        )

    @classmethod
    def gen_from_txt(cls, file_location, detector_config: DetectorConfig = None, skip_header: int = 2, transpose = False):
        # This is an impure function
        all_data = np.genfromtxt(file_location, skip_header = skip_header)
        rows, cols = all_data.shape
        fil_all_data = lambda column_index: np.zeros(rows) if column_index >= cols else all_data[:, column_index]
        intensity_col = all_data[:, 2]
        shadow_factor = fil_all_data(7)
        if not np.sum(shadow_factor**2):
            shadow_factor = intensity_col != 0
        data_dict = {
            QX : all_data[:, 1] if transpose else all_data[:, 0],
            QY : all_data[:, 0] if transpose else all_data[:, 1],
            INTENSITY : intensity_col,
            INTENSITY_ERROR : fil_all_data(3),
            QZ : fil_all_data(4),
            SIGMA_PARA : fil_all_data(5),
            SIGMA_PERP : fil_all_data(6),
            SHADOW_FACTOR : shadow_factor,
            SIMULATED_INTENSITY : fil_all_data(8),
            SIMUATED_INTENSITY_ERR : fil_all_data(9),
        }
        return cls.gen_from_data(data_dict=data_dict, detector_config=detector_config)

    @classmethod
    def gen_from_pandas(cls, dataframe: pd.DataFrame, detector_config: DetectorConfig | None = None):
        return DetectorImage(
            detector_pixels=[DetectorPixel.row_to_pixel(row.to_dict(), detector_config) for _, row in dataframe.iterrows()],
            polarization=detector_config.polarization if detector_config is not None else Polarization.UNPOLARIZED
        )

@array_cache
def make_smearing_function(pixel_list: Iterable[DetectorPixel], qx_matrix: np.ndarray, qy_matrix: np.ndarray, slicing_range: int | None = None) -> Callable[[np.ndarray], np.ndarray]:
    pixel_stuff = [(pixel.get_slicing_func(qx_matrix, qy_matrix, slicing_range), pixel.resolution_function(qx_matrix, qy_matrix)) for pixel in pixel_list]
    slicing_functions = [slicing_func for slicing_func, _ in pixel_stuff]
    big_resolution = np.array([slicing_func(resolution) for slicing_func, resolution in pixel_stuff])
    def smear(intensity: np.ndarray) -> np.ndarray:
        big_intensity = np.array([slicing_func(intensity) for slicing_func in slicing_functions])
        return np.sum(big_resolution * big_intensity, axis = (1,2))
    return smear

def subtract_detectors(detector_image: DetectorImage, subtraction: DetectorImage | float) -> DetectorImage:
    if isinstance(detector_image, float):
        return DetectorImage(
            detector_pixels=[
                subtract_background(pixel, detector_image)
                for pixel in self.detector_pixels
            ],
            polarization=self.polarization
        )
    return DetectorImage(
        detector_pixels=[
            subtract_nearest_pixel(pixel, detector_image.detector_pixels)
            for pixel in self.detector_pixels
        ],
        polarization=self.polarization
    )

if __name__ == "__main__":
    p = Polarization("spin_down")

#%%