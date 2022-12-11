#%%
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Tuple#List,

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import pandas as pd

from .array_cache import array_cache, method_array_cache
from .vector import Vector, broadcast_array_function#, dot
from .particles.particle import modulus_array 
from . import constants


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

def get_slicing_func_from_gaussian(gaussian: np.ndarray, slicing_range: int = 0) -> Callable[[np.ndarray], np.ndarray]:
    arg_of_max = np.where(gaussian == np.amax(gaussian)) 
    if slicing_range == 0:
        slicing_range = int(np.max(gaussian.shape) / DEFAULT_SLICING_FRACTION_DENOM)
    i_min = np.max([arg_of_max[0][0] - int(slicing_range / 2), 0])
    i_max = np.min([i_min + slicing_range, gaussian.shape[0] - 1])
    if i_max == gaussian.shape[0] - 1:
        i_min = i_max - slicing_range
    j_min = np.max([arg_of_max[1][0] - int(slicing_range / 2), 0])
    j_max = np.min([j_min + slicing_range, gaussian.shape[1] - 1])
    if j_max == gaussian.shape[1] - 1:
        j_min = j_max - slicing_range
    return lambda arr: arr[i_min:i_max, j_min:j_max]

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
        q = modulus_array(qx, qy)
        sigma_para = modulus_array(q * (self.wavelength_spread / 2), sigma_geom)
        return sigma_para


def orthogonal_xy(x: float, y: float) -> Tuple[float, float]: # If I find I need this function a lot, I'll make it a method in the Vector class, but for now I'm happy for it to be a helper function
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
    simulated_intensity: float = 0
    simulated_intensity_err: float = 0

    @property
    def q_vector(self) -> Vector:
        return Vector(self.qX, self.qY, self.qZ)

    def _resolution_function_calculator(self, qx_array: np.ndarray, qy_array: np.ndarray) -> np.ndarray:
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

    @method_array_cache
    def precompute_pixel_smearer(self, qx_array: np.ndarray, qy_array: np.ndarray, slicing_range: int = 0) -> Callable[[np.ndarray], np.ndarray]:
        resolution_function = self._resolution_function_calculator(qx_array, qy_array)
        slicing_func = get_slicing_func_from_gaussian(resolution_function, slicing_range=slicing_range)
        resolution_subset = slicing_func(resolution_function)# Leave this in the scope
        normalized_resolution_subset = resolution_subset / np.sum(resolution_subset)
        return lambda simulated_intensity : normalized_resolution_subset * slicing_func(simulated_intensity)

    def smear_pixel(self, simulated_intensity_array: np.ndarray, qx_array: np.ndarray, qy_array: np.ndarray, shadow_is_zero: bool = True, slicing_range: int = 0) -> None:
        if shadow_is_zero and not self.shadow_factor:
            self.simulated_intensity = 0
        else:
            pixel_smearer = self.precompute_pixel_smearer(qx_array, qy_array, slicing_range=slicing_range)
            self.simulated_intensity = np.sum(pixel_smearer(simulated_intensity_array))

    def to_dict(self):
        return {
            QX : self.qX,
            QY : self.qY,
            INTENSITY: self.intensity,
            INTENSITY_ERROR: self.intensity_err,
            QZ : self.qZ,
            SIGMA_PARA : self.sigma_para,
            SIGMA_PERP: self.sigma_perp,
            SHADOW_FACTOR : int(self.shadow_factor),
            SIMULATED_INTENSITY: self.simulated_intensity,
            SIMUATED_INTENSITY_ERR : self.simulated_intensity_err,
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
        simulated_intensity = data_row.get(SIMULATED_INTENSITY, 0)
        return cls(
            qX=qx_i,
            qY=qy_i,
            intensity=intensity,
            intensity_err=intensity_error,
            qZ=qZ,
            sigma_para=sigma_para,
            sigma_perp=sigma_perp,
            shadow_factor=shadow_factor,
            simulated_intensity=simulated_intensity,
            )


@dataclass
class DetectorImage: # Major refactor needed for detector image, as it shouldn't be responsible for how it's plotted
    _detector_pixels: np.ndarray
    polarization: Polarization = Polarization.UNPOLARIZED

    def array_from_pixels(self, pixel_func: Callable[[DetectorPixel], Any], output_dtype = np.float64) -> np.ndarray:
        broadcast_function = broadcast_array_function(pixel_func, output_dtype=output_dtype)
        return broadcast_function(self._detector_pixels)

    @property
    def qX(self) -> np.ndarray:
        get_qxi = lambda pixel: pixel.qX
        return self.array_from_pixels(get_qxi)

    @property
    def qY(self) -> np.ndarray:
        get_qyi = lambda pixel: pixel.qY
        return self.array_from_pixels(get_qyi)

    @property
    def q(self) -> np.ndarray:
        return modulus_array(self.qX, self.qY)

    @property
    def intensity(self) -> np.ndarray:
        get_intensity = lambda pixel: pixel.intensity
        return self.array_from_pixels(get_intensity)

    @intensity.setter
    def intensity(self, new_intensity: np.ndarray) -> None:
        def set_intensity(pixel: DetectorPixel, new_i: float):
            pixel.intensity = new_i
        set_intensity_matrix = np.frompyfunc(set_intensity, 2, 0)
        set_intensity_matrix(self._detector_pixels, new_intensity)

    @property
    def intensity_err(self) -> np.ndarray:
        get_intensity_err = lambda pixel: pixel.intensity_err
        return self.array_from_pixels(get_intensity_err)

    @intensity_err.setter
    def intensity_err(self, new_intensity_err: np.ndarray) -> None:
        def set_intensity_err(pixel: DetectorPixel, new_err: float):
            pixel.intensity_err = new_err
        set_intensity_err_matrix = np.frompyfunc(set_intensity_err, 2, 0)
        set_intensity_err_matrix(self._detector_pixels, new_intensity_err)

    @property
    def qZ(self) -> np.ndarray:
        get_qzi = lambda pixel: pixel.qZ
        return self.array_from_pixels(get_qzi)

    @property
    def sigma_para(self) -> np.ndarray:
        get_sigma_para = lambda pixel: pixel.sigma_para
        return self.array_from_pixels(get_sigma_para)

    @property
    def sigma_perp(self) -> np.ndarray:
        get_sigma_perp = lambda pixel: pixel.sigma_perp
        return self.array_from_pixels(get_sigma_perp)

    @property
    def shadow_factor(self) -> np.ndarray:
        get_shadow_factor = lambda pixel: pixel.shadow_factor
        return self.array_from_pixels(get_shadow_factor, output_dtype=bool)

    @shadow_factor.setter
    def shadow_factor(self, new_shadow_factor: np.ndarray) -> None:
        def set_shadow_factor(pixel: DetectorPixel, new_shadow: bool):
            pixel.shadow_factor = bool(new_shadow)
        set_intensity_err_matrix = np.frompyfunc(set_shadow_factor, 2, 0)
        set_intensity_err_matrix(self._detector_pixels, new_shadow_factor)

    def _q_delta(self, pixel_para_fn: Callable[[DetectorPixel], float], pixel_perp_fn: Callable[[DetectorPixel], float]) -> float:
        smallest_q_s = sorted(self._detector_pixels, key = lambda pixel : Vector(pixel_perp_fn(pixel), pixel_para_fn(pixel)).mag)[:9]
        next_qx = sorted(smallest_q_s[1:], key = lambda pixel : np.abs(pixel_perp_fn(pixel) - pixel_perp_fn(smallest_q_s[0])))[0]
        return np.abs(pixel_para_fn(next_qx) - pixel_para_fn(smallest_q_s[0]))

    @property
    def qx_delta(self) -> float:
        return self._q_delta(lambda pixel : pixel.qX, lambda pixel : pixel.qY)
        

    @property
    def qy_delta(self) -> float:
        return self._q_delta(lambda pixel : pixel.qY, lambda pixel : pixel.qX)
        

    @property
    def qxqy_delta(self) -> Tuple[float, float]:
        return self.qx_delta, self.qy_delta

    @staticmethod
    def plot_intensity_matrix(intensity_matrix, qx, qy , log_intensity = True, show_crosshair = True, levels = 30, cmap = 'jet', show_fig: bool = True) -> Figure:
        # Note to self, maybe replace this with pcolormesh
        fig, ax = plt.subplots()
        range_arr = lambda arr: np.max(arr) - np.min(arr)
        aspect_ratio = range_arr(qx) / range_arr(qy)
        fig.set_size_inches(5,5 / aspect_ratio)
        ax.contourf(qx, qy, np.log(intensity_matrix) if log_intensity else intensity_matrix, levels = levels, cmap = cmap)
        ax.set_xlabel(r'Q$_{x}$ ($\AA^{-1}$)',fontsize =  16)#'x-large')
        ax.set_ylabel(r'Q$_{y}$ ($\AA^{-1}$)',fontsize =  16)#'x-large')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        if show_crosshair:
            ax.axhline(0, linestyle='-', color='k') # horizontal lines
            ax.axvline(0, linestyle='-', color='k') # vertical lines
        ax.set_box_aspect(1/aspect_ratio)
        fig.tight_layout()
        if show_fig:
            plt.show()
        return fig

    def intensity_2d(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        qx_delta, qy_delta = self.qxqy_delta
        line_maker = lambda arr, step: np.arange(start = np.min(arr), stop = np.max(arr), step = step)
        qx_line = line_maker(self.qX, qx_delta / 2)
        qy_line = line_maker(self.qY, qy_delta / 2)
        qx, qy = np.meshgrid(qx_line, qy_line)
        tensities = qx * 0
        shadow = tensities * 0
        this_qX, this_qY, this_intensity, this_shadow = self.qX, self.qY, self.intensity, self.shadow_factor
        for idx, qx_i in np.ndenumerate(qx):
            qy_i = qy[idx]
            distances = np.sqrt((this_qX - qx_i)**2 + (this_qY - qy_i)**2)
            amin = np.argmin(distances)
            tensities[idx] = this_intensity[amin]
            shadow[idx] = this_shadow[amin]
        return qx, qy, tensities, shadow.astype(bool)

    def plot_intensity(self, log_intensity = True, show_crosshair = True, levels = 30, cmap = 'jet', show_fig: bool = True) -> Figure:
        qx, qy, tensities, shadow = self.intensity_2d()
        return type(self).plot_intensity_matrix(shadow * tensities, qx, qy, log_intensity=log_intensity, show_crosshair=show_crosshair, levels=levels, cmap = cmap, show_fig=show_fig)
    
    def get_pandas(self) -> pd.DataFrame:
        d = [pixel.to_dict() for _, pixel in np.ndenumerate(self._detector_pixels)]
        d[0].update({POLARIZATION : self.polarization.value})
        return pd.DataFrame(d)

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
    def gen_from_pandas(cls, dataframe: pd.DataFrame, detector_config: DetectorConfig = None):
        all_qx = dataframe[QX].astype(np.float64)
        all_qy = dataframe[QY].astype(np.float64)
        all_intensity = dataframe[INTENSITY].astype(np.float64)
        filter_optional_column = lambda col_title: dataframe.get(col_title, np.zeros(all_intensity.size)).astype(np.float64)
        all_intensity_error = filter_optional_column(INTENSITY_ERROR)
        all_qz = filter_optional_column(QZ)
        all_sigma_para = filter_optional_column(SIGMA_PARA)
        all_sigma_perp = filter_optional_column(SIGMA_PERP)
        all_shadow_factor = filter_optional_column(SHADOW_FACTOR)
        all_simulated_intensity = filter_optional_column(SIMULATED_INTENSITY)
        if not np.sum(all_shadow_factor**2):
            all_shadow_factor = all_intensity != 0
        data_dict = {
            QX : all_qx,
            QY : all_qy,
            INTENSITY : all_intensity,
            INTENSITY_ERROR : all_intensity_error,
            QZ : all_qz,
            SIGMA_PARA : all_sigma_para,
            SIGMA_PERP : all_sigma_perp,
            SHADOW_FACTOR : all_shadow_factor,
            SIMULATED_INTENSITY : all_simulated_intensity,
            POLARIZATION : dataframe.iloc[0].get(POLARIZATION, Polarization.UNPOLARIZED.value)
        }
        pass_config_on = np.sum(all_sigma_para**2) == 0 and np.sum(all_sigma_perp**2) == 0
        return cls.gen_from_data(data_dict=data_dict, detector_config=detector_config if pass_config_on else None)


@dataclass
class SimulatedDetectorImage(DetectorImage):
    
    @property
    def experimental_intensity(self) -> np.ndarray:
        return self.intensity

    @property
    def simulated_intensity(self) -> np.ndarray:
        get_simulated_intensity =  lambda pixel: pixel.simulated_intensity
        return self.array_from_pixels(get_simulated_intensity)

    @simulated_intensity.setter
    def simulated_intensity(self, new_simulated_intensity: np.ndarray) -> None:
        def set_intensity(pixel: DetectorPixel, new_i: float) -> None:
            pixel.simulated_intensity = new_i
        set_intensity_matrix = np.frompyfunc(set_intensity, 2, 0)
        set_intensity_matrix(self._detector_pixels, new_simulated_intensity)

    @property
    def simulated_intensity_err(self) -> np.ndarray:
        get_simulated_intensity_err = lambda pixel: pixel.simulated_intensity_err
        return self.array_from_pixels(get_simulated_intensity_err)

    def smear(self, intensity: np.ndarray, qx_array: np.ndarray, qy_array: np.ndarray, shadow_is_zero: bool = True) -> np.ndarray:
        def smear_pixel(pixel: DetectorPixel) -> float:
            pixel.smear_pixel(intensity, qx_array, qy_array, shadow_is_zero)
            return pixel.simulated_intensity
        return self.array_from_pixels(smear_pixel)

    def smear(self, intensity: np.ndarray, qx_array: np.ndarray, qy_array: np.ndarray, shadow_is_zero: bool = True) -> np.ndarray:
        zero_intensity = 0 * intensity
        dimension_getter = lambda arr: len(arr.shape)
        def get_slicing_function(pixel: DetectorPixel) -> Callable[[np.ndarray], np.ndarray]:
            slicing_func_name = f"_slicing_func_do_not_touch_{id(qx_array)}_{id(qy_array)}"
            if not hasattr(pixel, slicing_func_name):
                resolution = pixel._resolution_function_calculator(qx_array, qy_array)
                res_slicing_func = get_slicing_func_from_gaussian(resolution)
                setattr(pixel, slicing_func_name, res_slicing_func)
            return getattr(pixel, slicing_func_name)
        def slice_pixel_intensity(pixel: DetectorPixel) -> np.ndarray:
            slicing_function = get_slicing_function(pixel)
            return slicing_function(intensity if (pixel.shadow_factor or not shadow_is_zero) else zero_intensity)
        def slice_gauss(pixel: DetectorPixel) -> np.ndarray:
            resolution = pixel._resolution_function_calculator(qx_array, qy_array)
            slicing_function = get_slicing_function(pixel)
            return slicing_function(resolution)
        resolution_func_name = f"_sliced_resolution_do_not_touch_{id(qx_array)}_{id(qy_array)}" + ''.join(str(id(pixel)) for pixel in self._detector_pixels)
        if not hasattr(self, resolution_func_name):
            big_resolution_init = [slice_gauss(pixel) for pixel in self._detector_pixels]
            setattr(self, resolution_func_name, np.array(big_resolution_init))
        big_resolution = getattr(self, resolution_func_name)
        #pixel_intensity_slicer = np.frompyfunc(slice_pixel_intensity, nin = 1, nout = 1)
        big_intensity = np.array([slice_pixel_intensity(pixel) for pixel in self._detector_pixels])
        adding_axes = tuple(range(dimension_getter(self._detector_pixels), dimension_getter(big_intensity)))
        simulated_intensity = np.sum(big_resolution * big_intensity, axis = adding_axes)
        self.simulated_intensity = simulated_intensity
        return simulated_intensity
        '''def pixel_smear_and_intensity(pixel: DetectorPixel) -> Tuple[np.ndarray, np.ndarray]:
            sliced_resolution_name = f"_sliced_resolution_do_not_touch_{id(qx_array)}_{id(qy_array)}"
            slicing_func_name = f"_slicing_func_do_not_touch_{id(qx_array)}_{id(qy_array)}"
            if not all(hasattr(pixel, att) for att in (sliced_resolution_name, slicing_func_name)):
                resolution = pixel._resolution_function_calculator(qx_array, qy_array)
                res_slicing_func = get_slicing_func_from_gaussian(resolution)
                setattr(pixel, sliced_resolution_name, res_slicing_func(resolution))
                setattr(pixel, slicing_func_name, res_slicing_func)
            sliced_resolution = getattr(pixel, sliced_resolution_name)
            slicing_func = getattr(pixel, slicing_func_name)
            return sliced_resolution, slicing_func(intensity if (pixel.shadow_factor or not shadow_is_zero) else zero_intensity)
        pixel_smear_and_intensity_pyfunc = np.frompyfunc(pixel_smear_and_intensity, nin = 1, nout = 2)
        pixel_smear, pixel_intensity = pixel_smear_and_intensity_pyfunc(self._detector_pixels)
        big_resolution_arr, big_intensity_arr = np.array([p for p in pixel_smear]), np.array([b for b in pixel_intensity])
        adding_axes = tuple(range(dimension_getter(self._detector_pixels), dimension_getter(big_intensity_arr)))
        return np.sum(big_resolution_arr * big_intensity_arr, axis = adding_axes)'''
        
    def simulated_intensity_2d(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        qx, qy, _, shadow = self.intensity_2d()
        simulated_tensities = qx * 0
        this_qX, this_qY, this_simulated_intensity = self.qX, self.qY, self.simulated_intensity
        for idx, qx_i in np.ndenumerate(qx):
            qy_i = qy[idx]
            distances = np.sqrt((this_qX - qx_i)**2 + (this_qY - qy_i)**2)
            simulated_tensities[idx] = this_simulated_intensity[np.argmin(distances)]
        return qx, qy, simulated_tensities, shadow

    def experimental_simulated_2d(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        qx, qy, experimental_intensity, shadow = self.intensity_2d()
        simulated_tensities = qx * 0
        this_qX, this_qY, this_simulated_intensity = self.qX, self.qY, self.simulated_intensity
        for idx, qx_i in np.ndenumerate(qx):
            qy_i = qy[idx]
            if qx_i < 0:
                simulated_tensities[idx] = experimental_intensity[idx]
            else:
                distances = np.sqrt((this_qX - qx_i)**2 + (this_qY - qy_i)**2)
                simulated_tensities[idx] = this_simulated_intensity[np.argmin(distances)]
        return qx, qy, simulated_tensities, shadow

    def plot_intensity(self, mode: int = 2, log_intensity: bool = True, show_crosshair: bool = True, levels: int = 30, cmap: str = 'jet', show_fig: bool = True) -> Figure:
        intensity_matrix_maker = [
            self.intensity_2d,
            self.simulated_intensity_2d, #self.simulated_intensity,
            self.experimental_simulated_2d,
        ][mode]
        qx, qy, intensity, shadow = intensity_matrix_maker()
        return type(self).plot_intensity_matrix(shadow * intensity, qx, qy, log_intensity=log_intensity, show_crosshair=show_crosshair, levels = levels, cmap = cmap, show_fig=show_fig)




if __name__ == "__main__":
    pass

#%%