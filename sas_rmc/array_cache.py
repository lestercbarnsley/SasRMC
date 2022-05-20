#%%

from functools import wraps
from typing import Tuple
from enum import Enum

import numpy as np

from .vector import Vector

CLASS_MAX_SIZE = 18
MAX_SIZE = 50
DEFAULT_PRECISION = 14

immutable_types = (str, float, int, Enum, np.float64)

def round_vector(vector: Vector, precision: int = DEFAULT_PRECISION) -> Tuple[float, float, float]:
    round_vector_comp = lambda comp: round(comp, precision)
    return round_vector_comp(vector.x), round_vector_comp(vector.y), round_vector_comp(vector.z)

def pass_arg(arg):
    if type(arg) in immutable_types:
        return arg
    if isinstance(arg, np.ndarray) and arg.flags.writeable:
        arg.flags.writeable = False
    if isinstance(arg, Vector):
        return round_vector(arg)
    if isinstance(arg, list) or isinstance(arg, tuple):
        return tuple(pass_arg(a) for a in arg) # I finally found a use case for recursion
    if isinstance(arg, dict):
        return tuple((pass_arg(k), pass_arg(v)) for k, v in arg.items())
    return id(arg)

def array_cache(func, max_size: int = MAX_SIZE):
    cache = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        kwarg_tuple = pass_arg(kwargs)
        argument_tuple = pass_arg(args) + kwarg_tuple
        if argument_tuple not in cache:
            if len(cache) >= max_size:
                keys = list(cache.keys())
                uncached = [cache.pop(key) for key in keys[:-int(max_size / 2)]]
            result = func(*args, **kwargs)
            cache[argument_tuple] = result
        return cache[argument_tuple]
      
    return wrapper

def method_array_cache(func, max_size: int = CLASS_MAX_SIZE):
    cache = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        kwarg_tuple = tuple((k, pass_arg(v)) for k, v in kwargs.items())
        argument_tuple = tuple(pass_arg(arg) for arg in args) + kwarg_tuple
        object_id, cache_key = argument_tuple[0], argument_tuple[1:]
        if object_id not in cache:
            cache[object_id] = {}
        object_cache = cache[object_id]
        if cache_key not in object_cache:
            if len(object_cache) >= max_size:
                keys = list(object_cache.keys())
                uncached = [object_cache.pop(key) for key in keys[:-2]]
            result = func(*args, **kwargs)
            object_cache[cache_key] = result
        return object_cache[cache_key]
      
    return wrapper



if __name__ == "__main__":
    pass

#%%