#%%

from dataclasses import dataclass
from functools import wraps
from typing import Tuple
from enum import Enum

import numpy as np

from .vector import Vector

MAX_SIZE = 5

immutable_types = (str, float, int, Enum)

def round_vector(vector: Vector) -> Tuple[int, int, int]:
    round_vector_comp = lambda comp: int(comp * (2**20))
    return round_vector_comp(vector.x), round_vector_comp(vector.y), round_vector_comp(vector.z)

def pass_arg(arg):
    if type(arg) in immutable_types:
        return arg
    if isinstance(arg, np.ndarray) and arg.flags.writeable:
        arg.flags.writeable = False
    if isinstance(arg, Vector):
        return round_vector(arg)
    if isinstance(arg, list):
        return tuple((pass_arg(a) for a in arg))
    return id(arg)

def array_cache(func, max_size: int = MAX_SIZE):
    cache = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        kwarg_tuple = tuple((k, pass_arg(v)) for k, v in kwargs.items())
        argument_tuple = tuple(pass_arg(arg) for arg in args) + kwarg_tuple
        object_id = argument_tuple[0]
        if object_id not in cache:
            cache[object_id] = {}
        object_cache = cache[object_id]
        if argument_tuple not in object_cache:
            if len(object_cache) >= max_size:
                object_cache.pop(list(object_cache.keys())[0])
            result = func(*args, **kwargs)
            object_cache[argument_tuple] = result
        return object_cache[argument_tuple]
      
    return wrapper


@array_cache
def say(val):
    print(val)
    return val

@array_cache
def say_2(val):
    print(val)
    print(val)
    return 2 * val

@dataclass
class TestClass:
    a: float = 0

    @array_cache
    def show_a(self):
        print(self.a)
        return self.a


if __name__ == "__main__":
    pass

#%%