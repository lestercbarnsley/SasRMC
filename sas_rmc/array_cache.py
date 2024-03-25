#%%

from functools import wraps
from enum import Enum

import numpy as np

from sas_rmc import Vector

CLASS_MAX_SIZE = 18
MAX_SIZE = 50

immutable_types = (str, float, int, Enum, np.float64)

def create_arg_key(arg: tuple) -> tuple:
    if any(isinstance(arg, t) for t in [str, float, int, Enum, np.float64]):
        return arg
    if any(isinstance(arg, t) for t in [tuple, list, set]):
        return tuple(create_arg_key(a) for a in arg)
    if isinstance(arg, dict):
        return tuple((create_arg_key(k), create_arg_key(v)) for k, v in arg.items())
    if isinstance(arg, Vector):
        return tuple(create_arg_key(c) for c in arg.itercomps())
    if isinstance(arg, np.ndarray):
        if arg.flags.writeable:
            arg.flags.writeable = False
    return id(arg)

def create_function_cache_key(*args, **kwargs) -> tuple:
    return () + create_arg_key(args) + create_arg_key(kwargs)
    

def array_cache(func = None, max_size: int = None):
    max_size = max_size if max_size is not None else MAX_SIZE
    def _array_cache(func):
        cache = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            argument_tuple = create_function_cache_key(*args, **kwargs)
            if argument_tuple not in cache:
                if len(cache) >= max_size:
                    keys = list(cache.keys())
                    for key in keys[0:int(max_size / 2)]:
                        cache.pop(key)
                result = func(*args, **kwargs)
                cache[argument_tuple] = {
                    'result' : result,
                    'args' : args,
                    'kwargs' : kwargs
                }
            return cache[argument_tuple]['result']
        
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