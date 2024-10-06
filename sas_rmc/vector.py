#%%
from typing import Callable, Iterator, Type, overload, Sequence
from dataclasses import dataclass

import numpy as np
from typing_extensions import Self

from sas_rmc import constants
from sas_rmc.array_cache import array_cache

PI = constants.PI
rng = constants.RNG


@overload
def cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, ...]: ...

@overload
def cross(a: Sequence[np.ndarray | float], b: Sequence[np.ndarray | float]) -> tuple[np.ndarray, ...]: ...

def cross(a: Sequence[np.ndarray | float], b: Sequence[np.ndarray | float]) -> tuple[np.ndarray | float, ...]:
    ax, ay, az = a[0], a[1], a[2]
    bx, by, bz = b[0], b[1], b[2]
    cx = ay*bz-az*by
    cy = az*bx-ax*bz
    cz = ax*by-ay*bx
    return cx, cy, cz


@overload
def dot(a: Sequence[float], b: Sequence[float]) -> float: ...

@overload
def dot(a: Sequence[np.ndarray | float], b: Sequence[np.ndarray | float]) -> np.ndarray: ...

def dot(a: Sequence[np.ndarray | float], b: Sequence[np.ndarray | float]) -> np.ndarray | float:
    ax, ay = a[:2]
    az = a[2] if len(a) > 2 else 0
    bx, by = b[:2]
    bz = b[2] if len(b) > 2 else 0
    return (ax * bx) + (ay * by) + (az * bz)

def broadcast_array_function(getter_function: Callable[[object], object], output_dtype: Type = np.float64) -> Callable[[np.ndarray],np.ndarray]:
    numpy_ufunc = np.frompyfunc(getter_function, 1, 1)
    return lambda arr : numpy_ufunc(arr).astype(output_dtype)

def broadcast_to_numpy_array(object_array: np.ndarray, getter_function: Callable[[object], object], output_dtype: Type = np.float64) -> np.ndarray:
    array_function = broadcast_array_function(getter_function=getter_function, output_dtype=output_dtype)
    return array_function(object_array)

@array_cache(max_size=100_000)
def magnitude(*comps: float) -> float:
    return np.sqrt((np.array(comps)**2).sum())


@dataclass
class Vector:
    x: float
    y: float
    z: float = 0.0

    @property
    def mag(self) -> float:
        return magnitude(self.x, self.y, self.z)
        

    def itercomps(self) -> Iterator[float]: 
        yield self.x
        yield self.y
        yield self.z

    def to_list(self) -> list[float]:
        return [comp for comp in self.itercomps()]

    def to_numpy(self) -> np.ndarray:
        return np.array(self.to_list())

    def to_tuple(self) -> tuple[float, ...]:
        return tuple(self.itercomps())
    
    def __iter__(self) -> Iterator[float]:
        return self.itercomps()

    def __len__(self) -> int:
        return 3

    @classmethod
    def null_vector(cls) -> Self:
        return cls(0,0,0)

    def __add__(self, vector2: Self) -> Self:
        x = self.x + vector2.x
        y = self.y + vector2.y
        z = self.z + vector2.z
        return type(self)(x = x, y = y, z = z)
    
    @overload
    def __mul__(self, vector_or_scalar: float) -> Self: ...

    @overload
    def __mul__(self, vector_or_scalar: Self) -> float: ...

    def __mul__(self, vector_or_scalar: float | Self ) -> float | Self:
        if isinstance(vector_or_scalar, Vector):
            return dot(self.to_tuple(), vector_or_scalar.to_tuple())
        return type(self)(
            x = self.x * vector_or_scalar,
            y = self.y * vector_or_scalar,
            z = self.z * vector_or_scalar
            )

    def __rmul__(self, scalar: float) -> Self:
        return self * scalar

    def __sub__(self, vector2: Self) -> Self:
        return self + (-1 * vector2)

    def __truediv__(self, divisor: float) -> Self:
        return self * (1/divisor)

    def cross(self, vector2: Self) -> Self:
        x, y, z = cross(self.to_tuple(), vector2.to_tuple())
        return type(self)(x, y, z)

    @property
    def unit_vector(self) -> Self:
        if self.mag == 0:
            return type(self).null_vector()
        if self.mag == 1:
            return self
        else:
            return self / self.mag

    def distance_from_vector(self, vector: Self) -> float:
        return (self - vector).mag

    def copy(self) -> Self:
        return self + Vector.null_vector()

    def to_dict(self, vector_str: str | None = None) -> dict[str, float]:
        key_prefix = f"{vector_str}." if vector_str is not None else ""
        keys = [f"{key_prefix}{dimension}" for dimension in ["X", "Y", "Z"]]
        return {key: component for (key, component) in zip(keys, self.itercomps())}

    @classmethod
    def from_list(cls, l: list[float]):
        return cls(x = l[0], y = l[1], z = l[2])

    @classmethod
    def from_numpy(cls, arr):
        return cls(x = arr[0], y = arr[1], z = arr[2])

    @classmethod
    def from_dict(cls, d: dict, vector_str: str | None = None):
        key_prefix = f"{vector_str}." if vector_str is not None else ""
        keys = [f"{key_prefix}{dimension}" for dimension in ["X", "Y", "Z"]]
        return cls(
            x = d.get(keys[0], 0.0),
            y = d.get(keys[1], 0.0),
            z = d.get(keys[2], 0.0)
        )
    
    def project_to_xy(self) -> Self:
        return type(self)(x = self.x, y = self.y)

    def rotated_basis(self) -> tuple[Self, Self, Self]:
        unit_a = self.unit_vector
        mostly_orthogonal_basis = [-1 * type(self)(0,0,1), -1 * type(self)(1,0,0), -1 * type(self)(0,1,0)]
        mostly_orthog = mostly_orthogonal_basis[np.argmax(unit_a.to_numpy() ** 2)]
        unit_b = (unit_a.cross(mostly_orthog)).unit_vector
        unit_c = (unit_a.cross(unit_b)).unit_vector
        return unit_a, unit_b, unit_c

    @classmethod
    def random_vector(cls, length: float = 1.0) -> Self:
        random_numbers = rng.uniform(-1, 1, size=3)
        return length * (cls.from_numpy(random_numbers).unit_vector)

    @classmethod
    def xy_from_angle(cls, length: float = 1, angle: float = 0):
        return cls(x = length * np.cos(angle), y = length * np.sin(angle))

    @classmethod
    def random_vector_xy(cls, length: float = 1):
        random_angle = rng.uniform(low = -PI, high = +PI)
        return cls.xy_from_angle(length = length, angle = random_angle)

    @classmethod
    def random_normal_vector(cls, step_size:float = 1.0):
        random_numbers = rng.normal(loc = 0, scale = step_size, size = 3)
        return cls.from_numpy(random_numbers)


        




if __name__ == "__main__":
    pass



   #%%

