#%%

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from sas_rmc.array_cache import array_cache


@dataclass
class FieldDirection(ABC):
    @abstractmethod
    def form_set_from_field_direction(self, fn: np.ndarray, fmx: np.ndarray, fmy: np.ndarray, fmz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass


@dataclass
class FieldDirectionX(FieldDirection):
    def form_set_from_field_direction(self, fn: np.ndarray, fmx: np.ndarray, fmy: np.ndarray, fmz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return fn, fmx, fmy, fmz
    

@dataclass
class FieldDirectionY(FieldDirection):
    def form_set_from_field_direction(self, fn: np.ndarray, fmx: np.ndarray, fmy: np.ndarray, fmz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return fn, fmy, fmx, fmz
    

@dataclass
class FieldDirectionZ(FieldDirection):
    def form_set_from_field_direction(self, fn: np.ndarray, fmx: np.ndarray, fmy: np.ndarray, fmz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return fn, fmz, fmx, fmy


@dataclass
class Polarizer(ABC):
    @abstractmethod
    def intensity_polarization(self, fn: np.ndarray, fmx: np.ndarray, fmy: np.ndarray, fmz: np.ndarray) -> np.ndarray:
        pass


@array_cache(max_size=1_000)
def mod(arr: np.ndarray) -> np.ndarray:
    return np.real(arr * np.conj(arr))

def minus_minus(fn: np.ndarray, fm_para: np.ndarray, fm_perp_1: np.ndarray, fm_perp_2: np.ndarray) -> np.ndarray:
    return mod(fn + fm_para)

def plus_plus(fn: np.ndarray, fm_para: np.ndarray, fm_perp_1: np.ndarray, fm_perp_2: np.ndarray) -> np.ndarray:
    return mod(fn - fm_para)

def minus_plus(fn: np.ndarray, fm_para: np.ndarray, fm_perp_1: np.ndarray, fm_perp_2: np.ndarray) -> np.ndarray:
    return mod(-fm_perp_1 - 1j * fm_perp_2)

def plus_minus(fn: np.ndarray, fm_para: np.ndarray, fm_perp_1: np.ndarray, fm_perp_2: np.ndarray) -> np.ndarray:
    return mod(-fm_perp_1 + 1j * fm_perp_2)


@dataclass
class PolarizerMinusMinus(Polarizer):
    field_direction: FieldDirection

    def intensity_polarization(self, fn: np.ndarray, fmx: np.ndarray, fmy: np.ndarray, fmz: np.ndarray) -> np.ndarray:
        fn_actual, fm_para, fm_perp_1, fm_perp_2 = self.field_direction.form_set_from_field_direction(fn, fmx, fmy, fmz)
        return minus_minus(fn_actual, fm_para, fm_perp_1, fm_perp_2)
    

@dataclass
class PolarizerPlusPlus(Polarizer):
    field_direction: FieldDirection

    def intensity_polarization(self, fn: np.ndarray, fmx: np.ndarray, fmy: np.ndarray, fmz: np.ndarray) -> np.ndarray:
        fn_actual, fm_para, fm_perp_1, fm_perp_2 = self.field_direction.form_set_from_field_direction(fn, fmx, fmy, fmz)
        return plus_plus(fn_actual, fm_para, fm_perp_1, fm_perp_2)
    

@dataclass
class PolarizerMinusPlus(Polarizer):
    field_direction: FieldDirection

    def intensity_polarization(self, fn: np.ndarray, fmx: np.ndarray, fmy: np.ndarray, fmz: np.ndarray) -> np.ndarray:
        fn_actual, fm_para, fm_perp_1, fm_perp_2 = self.field_direction.form_set_from_field_direction(fn, fmx, fmy, fmz)
        return minus_plus(fn_actual, fm_para, fm_perp_1, fm_perp_2)
    

@dataclass
class PolarizerPlusMinus(Polarizer):
    field_direction: FieldDirection

    def intensity_polarization(self, fn: np.ndarray, fmx: np.ndarray, fmy: np.ndarray, fmz: np.ndarray) -> np.ndarray:
        fn_actual, fm_para, fm_perp_1, fm_perp_2 = self.field_direction.form_set_from_field_direction(fn, fmx, fmy, fmz)
        return plus_minus(fn_actual, fm_para, fm_perp_1, fm_perp_2)
    

@dataclass
class PolarizerSpinUp(Polarizer):
    field_direction: FieldDirection

    def intensity_polarization(self, fn: np.ndarray, fmx: np.ndarray, fmy: np.ndarray, fmz: np.ndarray) -> np.ndarray:
        pol_minus_minus = PolarizerMinusMinus(self.field_direction)
        pol_minus_plus = PolarizerMinusPlus(self.field_direction)
        return pol_minus_minus.intensity_polarization(fn, fmx, fmy, fmz) + pol_minus_plus.intensity_polarization(fn, fmx, fmy, fmz)
    

@dataclass
class PolarizerSpinDown(Polarizer):
    field_direction: FieldDirection

    def intensity_polarization(self, fn: np.ndarray, fmx: np.ndarray, fmy: np.ndarray, fmz: np.ndarray) -> np.ndarray:
        pol_plus_minus = PolarizerPlusMinus(self.field_direction)
        pol_plus_plus = PolarizerPlusPlus(self.field_direction)
        return pol_plus_minus.intensity_polarization(fn, fmx, fmy, fmz) + pol_plus_plus.intensity_polarization(fn, fmx, fmy, fmz)
    

@dataclass
class PolarizerUnpolarized(Polarizer):
    field_direction: FieldDirection

    def intensity_polarization(self, fn: np.ndarray, fmx: np.ndarray, fmy: np.ndarray, fmz: np.ndarray) -> np.ndarray:
        spin_up = PolarizerSpinUp(self.field_direction)
        spin_down = PolarizerSpinDown(self.field_direction)
        return (spin_up.intensity_polarization(fn, fmx, fmy, fmz) + spin_down.intensity_polarization(fn, fmx, fmy, fmz)) / 2
    
