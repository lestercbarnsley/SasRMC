from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, overload

import numpy as np
from numpy import typing as npt
from typing_extensions import Self

@dataclass
class ValueWithError(ABC):
    @abstractmethod
    def get_value(self) -> float | np.ndarray:
        pass

    @abstractmethod
    def get_error(self) -> float | np.ndarray:
        pass

T = TypeVar('T', float, npt.NDArray[np.floating])

def add_with_error(value_1: T, error_1: T, value_2: float | np.ndarray, error_2: float | np.ndarray) -> T:
    return value_1 + error_1


@dataclass
class ArrWithError2(ValueWithError):
    array: np.ndarray
    error: np.ndarray

    def get_value(self) -> np.ndarray:
        return self.array
    
    def get_error(self) -> np.ndarray:
        return self.error

    def __add__(self, other: float | np.ndarray | ValueWithError) -> Self:
        if not isinstance(other, type(self)):
            return type(self)(
                array = self.array + other,
                error = self.error
            )
        return type(self)(
            array = self.get_value() + other.get_value(),
            error = np.sqrt(self.get_error()**2 + other.get_error()**2)
        )
    

@dataclass
class ValWithError(ValueWithError):
    value: float
    error: float

    def get_value(self) -> float:
        return self.value
    
    def get_error(self) -> float | np.ndarray:
        return self.error
    
    @overload
    def __add__(self, other: ArrWithError2) -> ArrWithError2:
        pass

    @overload
    def __add__(self, other: Self) -> Self:
        pass

    def __add__(self, other: ArrWithError2 | Self | float) -> ArrWithError2 | Self:
        if isinstance(other, float | int):
            return self + ValWithError(other, 0.0)
        if isinstance(other, ArrWithError2):
            return ArrWithError2(
                array=other.get_value() + self.get_value(),
                error=np.sqrt(other.get_value()**2 + self.get_value()**2)
            )
        return type(self)(
            value = self.get_value() + other.get_value(),
            error = np.sqrt(self.get_value()**2 + other.get_value()**2)
        )
    

@dataclass
class ArrWithError(Generic[T]):
    value: T
    error: T

    def get_value(self) -> T:
        return self.value
    
    def get_error(self) -> T:
        return self.error

    def __add__(self, other: T | Self) -> Self:
        if not isinstance(other, type(self)):
            return type(self)(value = other + self.value, error = self.error)
        return type(self)(
            value = self.value + other.value,
            error = self.error + other.error
        )
    
if __name__ == "__main__":
    v = ArrWithError(3.0, 4.0) + ArrWithError(3.0, 4.0)
    v2 = ArrWithError(np.array([3,1,3]), np.array([5,1,1]))
    v3 = v2 + v2
    v1 = ArrWithError2(np.array([1,2,3]), np.array([3,2,1]))+ ValWithError(3.0, 1.0)
    v3 = v1 + ValWithError(3.0, 1.0)    