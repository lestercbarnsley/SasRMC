#%%
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Tuple#List,

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import pandas as pd

from .array_cache import method_array_cache
from .vector import Vector, broadcast_array_function#, dot
from .particle import modulus_array 


PI = np.pi
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
    '''closest_in_space_arr_fn = broadcast_array_function(closest_in_space)
    closest_arr = closest_in_space_arr_fn(arr)
    return closest_arr'''

def fuzzy_unique(arr: np.ndarray) -> np.ndarray:
    unique_arr = np.unique(arr)
    test_range = 4 * int(np.sqrt(arr.shape[0]))
    if unique_arr.shape[0] < test_range:
        return unique_arr

    test_space_maker = lambda num : np.linspace(np.min(arr), np.max(arr), num = num) if num !=0 else np.array([np.inf])
    for i in range(5, test_range):#_ in enumerate(first_guess):
        test_space = test_space_maker(i)
        closest_arr = test_uniques(test_space, arr)
        if not all(t in closest_arr for t in test_space):
            return test_space_maker(i-1)
    raise Exception("Unable to find enough unique values in Q-space to map detector to 2-D grid")
    


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
        area_to_radius = lambda area: np.sqrt(area / PI)
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

    # I'd rather calculate this differently, using a pyfunc, but this has been the fastest method I've come up with to do this calculation
    def _resolution_function_calculator(self, qx_array: np.ndarray, qy_array: np.ndarray) -> np.ndarray:
        qx_offset = qx_array - self.qX
        qy_offset = qy_array - self.qY
        q_offset = (qx_offset, qy_offset, 0)
        ''' def q_arr_fn(q_vec_comp: Vector): #I'd rather use a lambda than a closure, but there's no point in this instance
            q_unit = q_vec_comp.unit_vector
            return qx_offset * q_unit.x + qy_offset * q_unit.y'''
        q_vec = Vector(self.qX, self.qY)
        q_para = q_vec.unit_vector
        perp_angle = np.arctan2(q_para.y, q_para.x) + (PI/2)
        q_perp = Vector.xy_from_angle(angle=perp_angle)
        q_para_arr = q_para.unit_vector * q_offset#q_arr_fn(q_para)
        q_perp_arr = q_perp.unit_vector * q_offset
        gaussian = np.exp(-(1/2) * ((q_para_arr / self.sigma_para)**2 + (q_perp_arr / self.sigma_perp)**2) )

        return gaussian / np.sum(gaussian)

    @method_array_cache
    def precompute_pixel_smearer(self, qx_array: np.ndarray, qy_array: np.ndarray, slicing_range: int = 0) -> Callable[[np.ndarray], float]:
        resolution_function = self._resolution_function_calculator(qx_array, qy_array)
        slicing_func = get_slicing_func_from_gaussian(resolution_function, slicing_range=slicing_range)
        resolution_subset = slicing_func(resolution_function)# Leave this in the scope
        normalized_resolution_subset = resolution_subset / np.sum(resolution_subset)
        return lambda simulated_intensity : np.sum(normalized_resolution_subset * slicing_func(simulated_intensity))

    def smear_pixel(self, simulated_intensity_array: np.ndarray, qx_array: np.ndarray, qy_array: np.ndarray, shadow_is_zero: bool = True, slicing_range: int = 0) -> None:
        if shadow_is_zero and not self.shadow_factor:
            self.simulated_intensity = 0
        else:
            pixel_smearer = self.precompute_pixel_smearer(qx_array, qy_array, slicing_range=slicing_range)
            self.simulated_intensity = pixel_smearer(simulated_intensity_array)

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
        #get_element = lambda k, default_v = 0 : data_row[k] if k in data_row else default_v
        qx_i = data_row[QX]
        qy_i = data_row[QY]
        intensity = data_row[INTENSITY]
        intensity_error = data_row.get(INTENSITY_ERROR, 0)#get_element('intensity_error')
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
class DetectorImage:
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

    @property
    def qx_delta(self) -> float:
        return np.average(np.diff(self.qX[0,:]))

    @property
    def qy_delta(self) -> float:
        return np.average(np.diff(self.qY[:,0]))

    @property
    def qxqy_delta(self) -> Tuple[float, float]:
        return self.qx_delta, self.qy_delta

    @staticmethod
    def plot_intensity_matrix(intensity_matrix, qx, qy , log_intensity = True, show_crosshair = True, levels = 30, cmap = 'jet', show_fig: bool = True) -> Figure:
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

    def plot_intensity(self, log_intensity = True, show_crosshair = True, levels = 30, cmap = 'jet', show_fig: bool = True) -> Figure:
        return type(self).plot_intensity_matrix(self.intensity, self.qX, self.qY, log_intensity=log_intensity, show_crosshair=show_crosshair, levels=levels, cmap = cmap, show_fig=show_fig)
    
    def get_pandas(self) -> pd.DataFrame:
        d = [pixel.to_dict() for _, pixel in np.ndenumerate(self._detector_pixels)]
        d[0].update({POLARIZATION : self.polarization.value})
        return pd.DataFrame(d)

    @classmethod
    def gen_from_data(cls, data_dict: dict, detector_config: DetectorConfig = None):
        qX_data = data_dict[QX]
        qY_data = data_dict[QY]
        qX_1d = fuzzy_unique(qX_data)#.tolist() Ultimately, this should be a strategy somehow
        qY_1d = fuzzy_unique(qY_data)#.tolist()
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
            i = np.argmin(np.abs(qX_1d - qX))#qX_1d.index(qX)
            j = np.argmin(np.abs(qY_1d - qY))#qY_1d.index(qY)
            detector_pixels[j, i] = DetectorPixel.row_to_pixel(row, detector_config=detector_config)
        polarization = detector_config.polarization if detector_config else Polarization(data_dict.get(POLARIZATION, Polarization.UNPOLARIZED.value))
        return cls(
            _detector_pixels = detector_pixels,
            polarization = polarization
        )

    @classmethod
    def gen_from_txt(cls, file_location, detector_config: DetectorConfig = None, skip_header: int = 2, transpose = False):
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

    def plot_intensity(self, mode: int = 2, log_intensity: bool = True, show_crosshair: bool = True, levels: int = 30, cmap: str = 'jet', show_fig: bool = True) -> Figure:
        intensity_matrix_maker = [
            lambda : self.experimental_intensity,
            lambda : self.simulated_intensity,
            lambda : np.where(self.qX < 0, self.experimental_intensity, self.simulated_intensity)
        ][mode] # Always be lazy!
        return type(self).plot_intensity_matrix(intensity_matrix_maker(), self.qX, self.qY, log_intensity=log_intensity, show_crosshair=show_crosshair, levels = levels, cmap = cmap, show_fig=show_fig)




if __name__ == "__main__":
    pass

#%%