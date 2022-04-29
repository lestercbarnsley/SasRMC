#%%
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, List, Tuple

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


from .array_cache import array_cache
from .vector import Vector, broadcast_to_numpy_array
from .particle import modulus_array 


PI = np.pi
DEFAULT_SLICING_FRACTION_DENOM = 4 # This considers 6.25% of the detector (i.e, 1/4^2) per pixel, which is more than sufficient

#@array_cache
def get_slicing_func_from_gaussian(gaussian: np.ndarray, slicing_range: int = 0) -> Callable[[np.ndarray], np.ndarray]:
    arg_of_max = np.where(gaussian == np.amax(gaussian)) # I really don't understand this line of code but apparently it works
    slicing_range_use = slicing_range if slicing_range else int(np.max(gaussian.shape) / DEFAULT_SLICING_FRACTION_DENOM) 
    i_min = np.max([arg_of_max[0][0] - int(slicing_range_use / 2), 0])
    i_max = np.min([i_min + slicing_range_use, gaussian.shape[0] - 1])
    if i_max == gaussian.shape[0] - 1:
        i_min = i_max - slicing_range_use
    j_min = np.max([arg_of_max[1][0] - int(slicing_range_use / 2), 0])
    j_max = np.min([j_min + slicing_range_use, gaussian.shape[1] - 1])
    if j_max == gaussian.shape[1] - 1:
        j_min = j_max - slicing_range_use
    return lambda arr: arr[i_min:i_max, j_min:j_max]


class Polarization(Enum):
    UNPOLARIZED = auto()
    SPIN_UP = auto()
    SPIN_DOWN = auto()
    MINUS_MINUS = auto()
    MINUS_PLUS = auto()
    PLUS_MINUS = auto()
    PLUS_PLUS = auto()


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
    #@array_cache
    def _resolution_function_calculator(self, qx_array: np.ndarray, qy_array: np.ndarray) -> np.ndarray:
        qx_offset = qx_array - self.qX
        qy_offset = qy_array - self.qY
        def q_arr_fn(q_vec_comp: Vector): #I'd rather use a lambda than a closure, but there's no point in this instance
            q_unit = q_vec_comp.unit_vector
            return qx_offset * q_unit.x + qy_offset * q_unit.y
        q_vec = Vector(self.qX, self.qY)
        q_para = q_vec.unit_vector
        perp_angle = np.arctan2(q_para.y, q_para.x) + (PI/2)
        q_perp = Vector.xy_from_angle(angle=perp_angle)
        q_para_arr = q_arr_fn(q_para)
        q_perp_arr = q_arr_fn(q_perp)
        gaussian = np.exp(-(1/2) * ((q_para_arr / self.sigma_para)**2 + (q_perp_arr / self.sigma_perp)**2) )

        return gaussian / np.sum(gaussian)

    @array_cache
    def precompute_pixel_smearer(self, qx_array: np.ndarray, qy_array: np.ndarray, slicing_range: int = 0) -> Callable[[np.ndarray], float]:
        resolution_function = self._resolution_function_calculator(qx_array, qy_array)
        slicing_func = get_slicing_func_from_gaussian(resolution_function, slicing_range=slicing_range)
        resolution_subset = slicing_func(resolution_function) # Leave this in the scope
        return lambda simulated_intensity : np.sum(resolution_subset * slicing_func(simulated_intensity))

    def smear_pixel(self, simulated_intensity_array: np.ndarray, qx_array: np.ndarray, qy_array: np.ndarray, shadow_is_zero: bool = True, slicing_range: int = 0) -> None:
        if shadow_is_zero and not self.shadow_factor:
            self.simulated_intensity = 0
        else:
            pixel_smearer = self.precompute_pixel_smearer(qx_array, qy_array, slicing_range=slicing_range)
            self.simulated_intensity = pixel_smearer(simulated_intensity_array)

    def to_dict(self):
        return {
            'qX': self.qX,
            'qY': self.qY,
            'intensity': self.intensity,
            'intensity_err': self.intensity_err,
            'qZ': self.qZ,
            'sigma_para': self.sigma_para,
            'sigma_perp': self.sigma_perp,
            'shadow_factor': int(self.shadow_factor),
            'simulated_intensity': self.simulated_intensity,
            'simulated_intensity_err': self.simulated_intensity_err,
        }



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
class DetectorImage:
    _detector_pixels: np.ndarray
    polarization: Polarization = Polarization.UNPOLARIZED

    @property
    def qX(self) -> np.ndarray:
        get_qxi = lambda pixel: pixel.qX
        return broadcast_to_numpy_array(self._detector_pixels, get_qxi)

    @property
    def qY(self) -> np.ndarray:
        get_qyi = lambda pixel: pixel.qY
        return broadcast_to_numpy_array(self._detector_pixels, get_qyi)

    @property
    def q(self) -> np.ndarray:
        return modulus_array(self.qX, self.qY)

    @property
    def intensity(self) -> np.ndarray:
        get_intensity = lambda pixel: pixel.intensity
        return broadcast_to_numpy_array(self._detector_pixels, get_intensity)

    @intensity.setter
    def intensity(self, new_intensity: np.ndarray) -> None:
        def set_intensity(pixel: DetectorPixel, new_i: float):
            pixel.intensity = new_i
        set_intensity_matrix = np.frompyfunc(set_intensity, 2, 0)
        set_intensity_matrix(self._detector_pixels, new_intensity)

    @property
    def intensity_err(self) -> np.ndarray:
        get_intensity_err = lambda pixel: pixel.intensity_err
        return broadcast_to_numpy_array(self._detector_pixels, get_intensity_err)

    @property
    def qZ(self) -> np.ndarray:
        get_qzi = lambda pixel: pixel.qZ
        return broadcast_to_numpy_array(self._detector_pixels, get_qzi)

    @property
    def sigma_para(self) -> np.ndarray:
        get_sigma_para = lambda pixel: pixel.sigma_para
        return broadcast_to_numpy_array(self._detector_pixels, get_sigma_para)

    @property
    def sigma_perp(self) -> np.ndarray:
        get_sigma_perp = lambda pixel: pixel.sigma_perp
        return broadcast_to_numpy_array(self._detector_pixels, get_sigma_perp)

    @property
    def shadow_factor(self) -> np.ndarray:
        get_shadow_factor = lambda pixel: pixel.shadow_factor
        shadow_factor_function = np.frompyfunc(get_shadow_factor, 1, 1)
        return shadow_factor_function(self._detector_pixels).astype(bool)

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
    def plot_intensity_matrix(intensity_matrix, qx, qy , log_intensity = True, show_crosshair = True, levels = 30, cmap = 'jet') -> None:
        fig, ax = plt.subplots()
        range_arr = lambda arr: np.max(arr) - np.min(arr)
        aspect_ratio = range_arr(qx) / range_arr(qy)
        fig.set_size_inches(4,4 / aspect_ratio)
        ax.contourf(qx, qy, np.log(intensity_matrix) if log_intensity else intensity_matrix, levels = levels, cmap = cmap)
        ax.set_xlabel(r'Q$_{x}$ ($\AA^{-1}$)',fontsize =  16)#'x-large')
        ax.set_ylabel(r'Q$_{y}$ ($\AA^{-1}$)',fontsize =  16)#'x-large')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        if show_crosshair:
            ax.axhline(0, linestyle='-', color='k') # horizontal lines
            ax.axvline(0, linestyle='-', color='k') # vertical lines
        plt.show()

    def plot_intensity(self, log_intensity = True, show_crosshair = True, levels = 30, cmap = 'jet'):
        type(self).plot_intensity_matrix(self.intensity, self.qX, self.qY, log_intensity=log_intensity, show_crosshair=show_crosshair, levels=levels, cmap = cmap)
    
    def get_pandas(self):
        d = []
        for pixel in self._detector_pixels.flatten():
            d.append(pixel.to_dict())
        return pd.DataFrame(d)

    @classmethod
    def row_to_pixel(cls, data_row: dict, detector_config: DetectorConfig = None) -> DetectorPixel:
        get_element = lambda k, default_v = 0 : data_row[k] if k in data_row else default_v
        qx_i = data_row['qX']
        qy_i = data_row['qY']
        intensity = data_row['intensity']
        intensity_error = get_element('intensity_error')
        qZ = get_element('qZ')
        sigma_para = get_element('sigma_para') if detector_config is None else detector_config.get_sigma_parallel(qx_i, qy_i)
        sigma_perp = get_element('sigma_perp') if detector_config is None else detector_config.get_sigma_geometric()
        shadow_factor = bool(get_element('shadow_factor', default_v=bool(intensity)))
        return DetectorPixel(
            qX=qx_i,
            qY=qy_i,
            intensity=intensity,
            intensity_err=intensity_error,
            qZ=qZ,
            sigma_para=sigma_para,
            sigma_perp=sigma_perp,
            shadow_factor=shadow_factor
            )

    @classmethod
    def gen_from_data(cls, data_dict: dict, detector_config: DetectorConfig = None):
        qX_data = data_dict['qX']
        qY_data = data_dict['qY']
        qX_1d = np.unique(qX_data).tolist()
        qY_1d = np.unique(qY_data).tolist()
        qx, qy = np.meshgrid(qX_1d, qY_1d)
        blank_canvas_pixel = lambda qx_i, qy_i : DetectorPixel(
            qX=qx_i,
            qY=qy_i,
            intensity=0,
            intensity_err=0,
            sigma_para= 0 if detector_config is None else detector_config.get_sigma_parallel(qx_i, qy_i),
            sigma_perp= 0 if detector_config is None else detector_config.get_sigma_geometric(),
            shadow_factor=False
            )
        make_blank_canvas = np.frompyfunc(blank_canvas_pixel, 2, 1)
        detector_pixels = make_blank_canvas(qx, qy).astype(object)
        for index, _ in enumerate(qX_data):
            get_item = lambda k: data_dict[k][index]
            qX, qY = get_item('qX'), get_item('qY')
            row = {
                'qX': qX,
                'qY': qY,
                'intensity': get_item('intensity'),
                'intensity_error' : get_item('intensity_error'),
                'qZ': get_item('qZ'),
                'sigma_para': get_item('sigma_para'),
                'sigma_perp': get_item('sigma_perp'),
                'shadow_factor': get_item('shadow_factor')
            }
            i = qX_1d.index(qX)
            j = qY_1d.index(qY)
            detector_pixels[j, i] = cls.row_to_pixel(row, detector_config=detector_config)
        return cls(
            _detector_pixels = detector_pixels,
            polarization = detector_config.polarization if detector_config is not None else Polarization.UNPOLARIZED
        
        )



    @classmethod
    def gen_from_txt(cls, file_location, detector_config: DetectorConfig = None, skip_header: int = 2, transpose = False):
        all_data = np.genfromtxt(file_location, skip_header = skip_header)
        rows, cols = all_data.shape
        fil_all_data = lambda column_index: np.zeros(rows) if column_index >= cols else all_data[:, column_index]
        data_dict = {
            'qX' : all_data[:, 1] if transpose else all_data[:, 0],
            'qY' : all_data[:, 0] if transpose else all_data[:, 1],
            'intensity' : all_data[:, 2],
            'intensity_error' : fil_all_data(3),
            'qZ' : fil_all_data(4),
            'sigma_para' : fil_all_data(5),
            'sigma_perp' : fil_all_data(6),
            'shadow_factor' : fil_all_data(7)
        }
        return cls.gen_from_data(data_dict=data_dict, detector_config=detector_config)

    @classmethod
    def gen_from_pandas(cls, dataframe: pd.DataFrame, detector_config: DetectorConfig = None):
        all_qx = dataframe['qX'].astype(np.float64)
        all_qy = dataframe['qY'].astype(np.float64)
        all_intensity = dataframe['intensity'].astype(np.float64)
        column_titles = dataframe.columns
        shape = all_intensity.size
        filter_optional_column = lambda col_title: dataframe[col_title].astype(np.float64) if col_title in column_titles else np.zeros(shape)
        def filter_optional_column_possibilities(col_title_possibilities: List[str]):
            for col_title in col_title_possibilities:
                all_vals = filter_optional_column(col_title)
                if np.sum(all_vals**2):
                    return all_vals
            return all_vals
        all_intensity_error = filter_optional_column_possibilities(['intensity_error','intensity_err'])
        all_qz = filter_optional_column('qZ')
        all_sigma_para = filter_optional_column('sigma_para')
        all_sigma_perp = filter_optional_column('sigma_perp')
        all_shadow_factor = filter_optional_column('shadow_factor')
        data_dict = {
            'qX' : all_qx,
            'qY' : all_qy,
            'intensity' : all_intensity,
            'intensity_error' : all_intensity_error,
            'qZ' : all_qz,
            'sigma_para' : all_sigma_para,
            'sigma_perp' : all_sigma_perp,
            'shadow_factor' : all_shadow_factor
        }
        if np.sum(all_sigma_para**2) and np.sum(all_sigma_perp**2):
            return cls.gen_from_data(data_dict = data_dict) # Don't pass the detector config if sigma_para and sigma_perp are present
        return cls.gen_from_data(data_dict=data_dict, detector_config=detector_config)


@dataclass
class SimulatedDetectorImage(DetectorImage):
    
    @property
    def experimental_intensity(self) -> np.ndarray:
        return self.intensity

    @property
    def simulated_intensity(self) -> np.ndarray:
        get_simulated_intensity = lambda pixel: pixel.simulated_intensity
        return broadcast_to_numpy_array(self._detector_pixels, get_simulated_intensity)

    @simulated_intensity.setter
    def simulated_intensity(self, new_simulated_intensity: np.ndarray) -> None:
        def set_intensity(pixel: DetectorPixel, new_i: float) -> None:
            pixel.simulated_intensity = new_i
        set_intensity_matrix = np.frompyfunc(set_intensity, 2, 0)
        set_intensity_matrix(self._detector_pixels, new_simulated_intensity)

    @property
    def simulated_intensity_err(self) -> np.ndarray:
        get_simulated_intensity_err = lambda pixel: pixel.simulated_intensity_err
        return broadcast_to_numpy_array(self._detector_pixels, get_simulated_intensity_err)

    def smear(self, intensity: np.ndarray, qx_array: np.ndarray, qy_array: np.ndarray, shadow_is_zero: bool = True) -> np.ndarray:
        def smear_pixel(pixel: DetectorPixel) -> None:
            pixel.smear_pixel(intensity, qx_array, qy_array, shadow_is_zero)
        smeared_intensity = np.frompyfunc(smear_pixel, 1, 0)
        smeared_intensity(self._detector_pixels)
        return self.simulated_intensity

    def plot_intensity(self, mode: int = 2, log_intensity: bool = True, show_crosshair: bool = True, levels: int = 30, cmap: str = 'jet'):
        intensity_matrix = {
            0 : self.experimental_intensity,
            1 : self.simulated_intensity,
            2 : np.where(self.qX < 0, self.experimental_intensity, self.simulated_intensity)
        }[mode]
        type(self).plot_intensity_matrix(intensity_matrix, self.qX, self.qY, log_intensity=log_intensity, show_crosshair=show_crosshair, levels = levels, cmap = cmap)

  

if __name__ == "__main__":
    pass

#%%