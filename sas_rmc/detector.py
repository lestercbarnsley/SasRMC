#%%
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable

import numpy as np
from numpy import typing as npt
from typing_extensions import Self

from sas_rmc import vector, Vector
from sas_rmc.array_cache import method_array_cache
from sas_rmc import constants


PI = constants.PI
DEFAULT_GAUSSIAN_FLOOR_FRACTION = 1e-4

# name string constants, this is the source of truth for the names of these quantities
QX = 'qX'
QY = 'qY'
INTENSITY = 'intensity'
INTENSITY_ERROR = 'intensity_err'
QZ = 'qZ'
SIGMA_PARA = 'sigma_para'
SIGMA_PERP = 'sigma_perp'
SHADOW_FACTOR = 'shadow_factor'
SIMULATED_INTENSITY = 'simulated_intensity'
SIMUATED_INTENSITY_ERR = 'simulated_intensity_err'
POLARIZATION = "Polarization"


class Polarization(Enum):
    UNPOLARIZED = "unpolarized"
    SPIN_UP = "spin_up"
    SPIN_DOWN = "spin_down"
    MINUS_MINUS = "minus_minus"
    MINUS_PLUS = "minus_plus"
    PLUS_MINUS = "plus_minus"
    PLUS_PLUS = "plus_plus"


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
    
    @method_array_cache
    def resolution_function(self, qx_array: np.ndarray, qy_array: np.ndarray) -> np.ndarray:
        qx_offset = qx_array - self.qX
        qy_offset = qy_array - self.qY
        qx = qx_array[0]
        q_offset = (qx_offset, qy_offset, np.array([0.0]))
        q_vec = Vector(self.qX, self.qY)
        q_para = q_vec.unit_vector
        q_perp = Vector(*orthogonal_xy(q_para.x, q_para.y))
        q_para_arr = vector.dot(q_para.unit_vector.to_tuple() , q_offset)
        q_perp_arr = vector.dot(q_perp.unit_vector.to_tuple(),  q_offset)
        gaussian = np.exp(-(1/2) * ((q_para_arr / self.sigma_para)**2 + (q_perp_arr / self.sigma_perp)**2) )

        return gaussian / np.sum(gaussian)
    
    @method_array_cache
    def get_smearing_func(self, qx_array: np.ndarray, qy_array: np.ndarray, gaussian_floor: float = DEFAULT_GAUSSIAN_FLOOR_FRACTION) -> Callable[[np.ndarray], float]:
        gaussian = self.resolution_function(qx_array, qy_array)
        idxs = np.where(gaussian > gaussian.max() * gaussian_floor)
        def slicing_func(arr: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            return arr[idxs]
        sliced_gaussian = slicing_func(gaussian)
        sliced_gaussian_sum = np.sum(sliced_gaussian).item()
        def smearing_func(arr: np.ndarray) -> float:
            return np.sum(slicing_func(arr) * sliced_gaussian).item() / sliced_gaussian_sum # This is way faster than np.average
            
        return smearing_func

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
    
    def subtract(self, pixel_2: Self) -> Self:
        if (self.qX != pixel_2.qX) or (self.qY != pixel_2.qY):
            raise ValueError("Cannot subtract two pixels that have different Q values")
        return type(self)(
            qX = self.qX,
            qY = self.qY,
            intensity = self.intensity - pixel_2.intensity,
            intensity_err= np.sqrt(self.intensity_err**2 + pixel_2.intensity_err**2),
            qZ = self.qZ,
            sigma_para=self.sigma_para,
            sigma_perp=self.sigma_perp,
            shadow_factor=self.shadow_factor
        )


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
    def distance_from_pixel(pixel: DetectorPixel) -> float:
        return (qx - pixel.qX)**2 + (qy - pixel.qY)**2
    return min(pixels, key = distance_from_pixel)

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
class DetectorImage: 
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
    

def make_smearing_function(pixel_list: Iterable[DetectorPixel], qx_matrix: np.ndarray, qy_matrix: np.ndarray, gaussian_floor: float = DEFAULT_GAUSSIAN_FLOOR_FRACTION) -> Callable[[np.ndarray], np.ndarray]:
    slicing_funcs = [pixel.get_smearing_func(qx_matrix, qy_matrix, gaussian_floor) for pixel in pixel_list]
    def smear(intensity: np.ndarray) -> np.ndarray:
        return np.array([slicing_func(intensity) for slicing_func in slicing_funcs])
    return smear

def subtract_two_detectors(detector_image_1: DetectorImage, detector_image_2: DetectorImage) ->  DetectorImage:
    if np.any(detector_image_1.qX != detector_image_2.qX) or np.any(detector_image_1.qY != detector_image_2.qY):
        raise ValueError("Subtraction not possible on two detectors with different q values")
    return DetectorImage(
        detector_pixels=[pixel_1.subtract(pixel_2) for pixel_1, pixel_2 in zip(detector_image_1.detector_pixels, detector_image_2.detector_pixels)],
        polarization=detector_image_1.polarization
    )

def subtract_flat_background(detector_image: DetectorImage, background: float) -> DetectorImage:
    return DetectorImage(
        detector_pixels=[
            subtract_background(pixel, background) for pixel in detector_image.detector_pixels
        ],
        polarization=detector_image.polarization
    )

def subtract_detectors(detector_image: DetectorImage, subtraction: DetectorImage | float) -> DetectorImage:
    if isinstance(subtraction, DetectorImage):
        return subtract_two_detectors(detector_image, subtraction)
    return subtract_flat_background(detector_image, subtraction)

if __name__ == "__main__":
    p = Polarization("spin_down")

    from numpy import typing as npt
    from typing import TypeVar

    T = TypeVar('T', bound=np.number)

    def show_numpy(arr: npt.NDArray[T]) -> T:
        a = arr.item(0)
        return a
        #return np.sum(arr)
    
    def get_item(l: list[T]) -> T:
        return l[0]
    
    a = show_numpy(np.array([3]))

    print(a)
    print(type(a))


#%%