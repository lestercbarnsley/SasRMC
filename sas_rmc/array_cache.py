#%%

from functools import wraps
from typing import Tuple
from enum import Enum

import numpy as np

from .vector import Vector

CLASS_MAX_SIZE = 18
MAX_SIZE = 50
DEFAULT_PRECISION = 8

immutable_types = (str, float, int, Enum, np.float64)

def _round_vector_comp(comp: float, precision: float):
    if np.abs(comp) < 1e-16:
        return 0.0
    for i in range(34):
        if np.abs(comp * 10**i) > 1:
            return int(comp * 10**(precision + i)) / 10**(precision + i)

def round_vector(vector: Vector, precision: int = DEFAULT_PRECISION) -> Tuple[float, float, float]:
    return tuple(_round_vector_comp(comp, precision) for comp in vector.itercomps())

def pass_arg(arg):
    if type(arg) in immutable_types:
        return arg
    if isinstance(arg, np.ndarray) and arg.flags.writeable:
        arg.flags.writeable = False
    if isinstance(arg, Vector):
        return round_vector(arg)
    if isinstance(arg, list) or isinstance(arg, tuple):
        return tuple(pass_arg(a) for a in arg)
    if isinstance(arg, dict):
        return tuple((pass_arg(k), pass_arg(v)) for k, v in arg.items())
    return id(arg)

def array_cache(func = None, max_size: int = None):
    if max_size is None:
        max_size = MAX_SIZE
    def _array_cache(func):
        cache = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            kwarg_tuple = pass_arg(kwargs)
            argument_tuple = pass_arg(args) + kwarg_tuple
            if argument_tuple not in cache:
                if len(cache) >= max_size:
                    keys = list(cache.keys())
                    uncached = [cache.pop(key) for key in keys[-int(max_size / 2):]]
                result = func(*args, **kwargs)
                cache[argument_tuple] = result
            return cache[argument_tuple]
        
        return wrapper
    if func is not None:
        return _array_cache(func)
    return _array_cache

def method_array_cache(func = None, max_size: int = None, cache_holder_keysize: int = None):
    if max_size is None:
        max_size = CLASS_MAX_SIZE
    if cache_holder_keysize is None:
        cache_holder_keysize = 1
    def _method_array_cache(func):
        cache = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_holder_items, other_args =  args[:cache_holder_keysize], args[cache_holder_keysize:]
            object_id = pass_arg(cache_holder_items)
            kwarg_tuple = tuple((k, pass_arg(v)) for k, v in kwargs.items())
            argument_tuple = tuple(pass_arg(arg) for arg in other_args) + kwarg_tuple
            #object_id, cache_key = argument_tuple[0], argument_tuple[1:]
            if object_id not in cache:
                cache[object_id] = {}
            object_cache = cache[object_id]
            if argument_tuple not in object_cache:
                if len(object_cache) >= max_size:
                    keys = list(object_cache.keys())
                    uncached = [object_cache.pop(key) for key in keys[-2:]]
                result = func(*args, **kwargs)
                object_cache[argument_tuple] = result
            return object_cache[argument_tuple]
        
        return wrapper
    if func is not None:
        return _method_array_cache(func)
    return _method_array_cache



if __name__ == "__main__":
    pass

#%%