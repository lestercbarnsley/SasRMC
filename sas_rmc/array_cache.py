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
    if isinstance(arg, np.ndarray):
        if arg.flags.writeable:
            arg.flags.writeable = False
        return id(arg)
    if isinstance(arg, Vector):
        return round_vector(arg)
    if isinstance(arg, list) or isinstance(arg, tuple):
        return tuple(pass_arg(a) for a in arg)
    if isinstance(arg, dict):
        return tuple((pass_arg(k), pass_arg(v)) for k, v in arg.items())
    return id(arg)

def array_cache(func = None, max_size: int = None):
    max_size = max_size if max_size is not None else MAX_SIZE
    def _array_cache(func):
        cache = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            kwarg_tuple = pass_arg(kwargs)
            argument_tuple = pass_arg(args) + kwarg_tuple
            if argument_tuple not in cache:
                if len(cache) >= max_size:
                    keys = list(cache.keys())
                    #uncached = [cache.pop(key) for key in keys[-int(max_size / 2):]]
                    uncached = [cache.pop(key) for key in keys[0:int(max_size / 2)]]
                result = func(*args, **kwargs)
                cache[argument_tuple] = result
            return cache[argument_tuple]
        
        return wrapper
    if func is not None:
        return _array_cache(func)
    return _array_cache

key_cache_avoider = ''.join(np.random.choice(['dsfa','ewrf','werfj','gjowq','glks','fjkds','jtks','dsaa','jgkda','ewiq']) for _ in range(10))

def method_array_cache(func = None, max_size: int = CLASS_MAX_SIZE, cache_holder_index: int = 0):
    def _method_array_cache(func):
        cache_name = f'_method_cache_{func.__name__}_{key_cache_avoider}_'

        @wraps(func)
        def wrapper(*args, **kwargs):
            obj = args[cache_holder_index]
            other_args = [a for i, a in enumerate(args) if i!=cache_holder_index]
            kwarg_tuple = tuple((k, pass_arg(v)) for k, v in kwargs.items())
            argument_tuple = tuple(pass_arg(arg) for arg in other_args) + kwarg_tuple
            if not hasattr(obj, cache_name):
                setattr(obj, cache_name, {})
            object_cache = getattr(obj, cache_name)
            if argument_tuple not in object_cache:
                if len(object_cache) >= max_size:
                    keys = list(object_cache.keys())
                    #uncached = [object_cache.pop(key) for key in keys[-2:]]
                    uncached = [object_cache.pop(key) for key in keys[0:-2]]
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