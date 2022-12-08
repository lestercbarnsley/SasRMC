#%%
from functools import reduce
from typing import Any, Callable, Generator, List, Tuple, Type
from dataclasses import dataclass
import math

import numpy as np
from . import constants

PI = constants.PI
rng = constants.RNG

def cross(a: Tuple, b: Tuple) -> Tuple:
    ax, ay, az = a[0], a[1], a[2]
    bx, by, bz = b[0], b[1], b[2]
    cx = ay*bz-az*by
    cy = az*bx-ax*bz
    cz = ax*by-ay*bx
    return cx, cy, cz

def dot(a: Tuple, b: Tuple) -> Any:
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

def composite_function(*func):
    compose = lambda f, g : lambda x : f(g(x))
    return reduce(compose, func, lambda x: x)


@dataclass
class Vector:
    x: float
    y: float
    z: float = 0.0

    @property
    def mag(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def itercomps(self) -> Generator: 
        yield self.x
        yield self.y
        yield self.z

    def to_list(self) -> list:
        return list(self.itercomps())#[self.x, self.y, self.z]

    def to_numpy(self) -> np.ndarray:
        return np.array(self.to_list())

    def to_tuple(self) -> Tuple[float, float, float]:
        return tuple(self.itercomps())

    def __len__(self) -> int:
        return 3

    @classmethod
    def null_vector(cls):
        return cls(0,0,0)

    def __add__(self, vector2):
        x = self.x + vector2.x
        y = self.y + vector2.y
        z = self.z + vector2.z
        return type(self)(x = x, y = y, z = z)

    def dot(self, vector_or_tuple) -> float:
        vector_as_tuple = vector_or_tuple.to_tuple() if isinstance(vector_or_tuple, Vector) else vector_or_tuple
        return dot(self.to_tuple(), vector_as_tuple)

    def __mul__(self, vector_or_scalar):
        if isinstance(vector_or_scalar, Vector) or type(vector_or_scalar) in [list, tuple]:#This should be hardcodes as the base class because a sub class should be multipliable by any Vector
            return self.dot(vector_or_scalar)
        return type(self)(
            x = self.x * vector_or_scalar,
            y = self.y * vector_or_scalar,
            z = self.z * vector_or_scalar
            )

    def __rmul__(self, scalar: float):
        return self * scalar

    def __sub__(self, vector2):
        return self + (-1 * vector2)

    def __truediv__(self, divisor: float):
        return self * (1/divisor)

    def cross(self, vector2):
        x, y, z = cross(self.to_tuple(), vector2.to_tuple())
        return type(self)(x, y, z)

    @property
    def unit_vector(self):
        if self.mag == 0:
            return type(self).null_vector()
        if self.mag == 1:
            return self
        else:
            return self / self.mag

    def distance_from_vector(self, vector) -> float:
        return (self - vector).mag

    def copy(self):
        return type(self)(
            x = self.x + 0.0,
            y = self.y + 0.0,
            z = self.z + 0.0
            )

    def to_dict(self, vector_str: str = None) -> dict:
        key_prefix = f"{vector_str}." if vector_str is not None else ""
        keys = [f"{key_prefix}{dimension}" for dimension in ["X", "Y", "Z"]]
        return {key: component for (key, component) in zip(keys, self.itercomps())}
        '''if vector_str is None:
            return {
                "X": self.x,
                "Y": self.y,
                "Z": self.z
            }
        return {
            f"{vector_str}.X" : self.x,
            f"{vector_str}.Y" : self.y,
            f"{vector_str}.Z" : self.z,
        }'''

    @classmethod
    def from_list(cls, l: List[float]):
        return cls(x = l[0], y = l[1], z = l[2])

    @classmethod
    def from_numpy(cls, arr):
        return cls(x = arr[0], y = arr[1], z = arr[2])

    @classmethod
    def from_dict(cls, d: dict, vector_str: str = None):
        key_prefix = f"{vector_str}." if vector_str is not None else ""
        keys = [f"{key_prefix}{dimension}" for dimension in ["X", "Y", "Z"]]
        return cls(
            x = d.get(keys[0], 0.0),
            y = d.get(keys[1], 0.0),
            z = d.get(keys[2], 0.0)
        )
        '''
        if vector_str is None:
            return cls(
                x = d.get('X', 0.0),
                y = d.get('Y', 0.0),
                z = d.get('Z', 0.0)
            )
        return cls(
            x = d.get(f'{vector_str}.X', 0.0),
            y = d.get(f'{vector_str}.Y', 0.0),
            z = d.get(f'{vector_str}.Z', 0.0)
        )'''

    def rotated_basis(self) -> Tuple:
        unit_a = self.unit_vector
        mostly_orthogonal_basis = [-1 * Vector(0,0,1), -1 * Vector(1,0,0), -1 * Vector(0,1,0)]
        mostly_orthog = mostly_orthogonal_basis[np.argmax(unit_a.to_numpy() ** 2)]
        unit_b = (unit_a.cross(mostly_orthog)).unit_vector
        unit_c = (unit_a.cross(unit_b)).unit_vector
        return unit_a, unit_b, unit_c

    @classmethod
    def random_vector(cls, length: float = 1.0):
        random_numbers = rng.uniform(-1, 1, size=3)
        return length * (cls.from_numpy(random_numbers).unit_vector)

    @classmethod
    def xy_from_angle(cls, length = 1, angle = 0):
        return cls(x = length * np.cos(angle), y = length * np.sin(angle))

    @classmethod
    def random_vector_xy(cls, length: float = 1):
        random_angle = rng.uniform(low = -PI, high = +PI)
        return cls.xy_from_angle(length = length, angle = random_angle)#length * cls(x = np.cos(random_angle), y = np.sin(random_angle))

    @classmethod
    def random_normal_vector(cls, step_size=1):
        random_numbers = rng.normal(loc = 0, scale = step_size, size = 3)
        return cls.from_numpy(random_numbers)


@dataclass
class VectorElement:
    position: Vector = Vector.null_vector()
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0

    @property
    def volume(self) -> float:
        return self.dx * self.dy * self.dz
        

@dataclass
class VectorSpace:
    vector_elements: np.ndarray

    def array_from_elements(self, element_function: Callable[[VectorElement], float], output_type = np.float64) -> np.ndarray:
        array_function = broadcast_array_function(element_function, output_dtype=output_type)
        return array_function(self.vector_elements)

    def field_from_element(self, field_function: Callable[[VectorElement], Vector], output_type = object) -> np.ndarray:
        field_function_arr = broadcast_array_function(field_function, output_dtype=output_type)
        return field_function_arr(self.vector_elements)
        
    @property
    def position(self) -> np.ndarray:
        get_position = lambda element: element.position
        return self.array_from_elements(get_position, object)

    @property
    def volume(self) -> np.ndarray:
        get_volume = lambda element: element.volume
        return self.array_from_elements(get_volume)

    @property
    def x(self) -> np.ndarray:
        get_x = lambda element: element.position.x
        return self.array_from_elements(get_x)

    @property
    def y(self) -> np.ndarray:
        get_y = lambda element: element.position.y
        return self.array_from_elements(get_y)

    @property
    def z(self) -> np.ndarray:
        get_z = lambda element: element.position.z
        return self.array_from_elements(get_z)

    @property
    def dx(self) -> np.ndarray:
        get_dx = lambda element: element.dx
        return self.array_from_elements(get_dx)

    @property
    def dy(self) -> np.ndarray:
        get_dy = lambda element: element.dy
        return self.array_from_elements(get_dy)

    @property
    def dz(self) -> np.ndarray:
        get_dz = lambda element: element.dz
        return self.array_from_elements(get_dz)

    def __getitem__(self, indices) -> VectorElement:
        i, j, k = indices
        return self.vector_elements[i, j, k]

    def change_position(self, vector_offset: Vector):
        def change_element_position(element: VectorElement):
            return VectorElement(
                position=element.position + vector_offset,
                dx = element.dx,
                dy = element.dy,
                dz = element.dz
            )
        element_space_maker = np.frompyfunc(change_element_position, 1, 1)
        return VectorSpace(
            vector_elements = element_space_maker(self.vector_elements)
        )

    @classmethod
    def gen_from_bounds(cls, x_min, x_max, x_num, y_min, y_max, y_num, z_min, z_max, z_num):
        x = np.linspace(x_min, x_max, num = x_num)
        y = np.linspace(y_min, y_max, num = y_num)
        z = np.linspace(z_min, z_max, num = z_num)
        dx = np.gradient(x)
        dy = np.gradient(y)
        dz = np.gradient(z)
        elements = [[[VectorElement(
            position=Vector(xi, yi, zi),
            dx=dxi,
            dy=dyi,
            dz=dzi
        ) for (dzi, zi) in zip(dz, z)]
        for (dyi, yi) in zip(dy, y)]
        for (dxi, xi) in zip(dx,x)]
        return cls(vector_elements = np.array(elements))


@dataclass
class Interface:
    position_marker: Vector = Vector.null_vector()
    normal: Vector = Vector.null_vector()

    def is_inside(self, position: Vector):
        return (position - self.position_marker) * self.normal < 0

    def on_surface(self, position: Vector):
        return (position - self.position_marker) * self.normal == 0

    def project_onto_surface(self, position: Vector):
        position_ref = position - self.position_marker
        return position_ref - (self.normal.unit_vector * position_ref) * self.normal.unit_vector + self.position_marker


if __name__ == "__main__":
    pass



#%%

