#%%

from functools import wraps
from enum import Enum
from typing import Callable, ParamSpec, TypeVar, MutableMapping, Hashable, Iterable, Any, overload
import string

import numpy as np

CLASS_MAX_SIZE = 18
MAX_SIZE = 50

immutable_types = (str, float, int, Enum, np.float64)

T = TypeVar("T")


@overload
def create_arg_key(arg: Iterable) -> tuple: ...

@overload
def create_arg_key(arg: MutableMapping) -> tuple: ...

def create_arg_key(arg: Any) -> Hashable:
    if hasattr(arg, 'items') and hasattr(arg, 'values') and hasattr(arg, 'keys'):
        return tuple((create_arg_key(k), create_arg_key(v)) for k, v in arg.items())
    if hasattr(arg, 'flags') and hasattr(arg.flags, 'writeable'):
        if arg.flags.writeable:
            arg.flags.writeable = False
        return id(arg)
    if isinstance(arg, str):
        return arg
    if hasattr(arg, '__iter__'):
        return tuple(create_arg_key(a) for a in arg)
    if hasattr(arg, '__hash__') and arg.__hash__ is not None:
        return arg
    return id(arg)

def create_function_cache_key(*args, **kwargs) -> Hashable:
    return () + create_arg_key(args) + create_arg_key(kwargs)

R = TypeVar("R")
P = ParamSpec("P")

def array_cache(func: Callable[P, R] | None = None, max_size: int | None = None) -> Callable[P, R]:
    max_size = max_size if max_size is not None else MAX_SIZE
    def _array_cache(func: Callable[P, R]) -> Callable[P, R]:
        cache = {}

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            argument_tuple = create_function_cache_key(*args, **kwargs)
            if argument_tuple not in cache:
                if len(cache) >= max_size:
                    half_keys = list(cache.keys())[0:int(max_size / 2)]
                    uncached = [cache.pop(key) for key in half_keys]
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

key_cache_avoider = ''.join(np.random.choice([c for c in string.ascii_letters + string.digits]) for _ in range(40))

def method_array_cache(func: Callable[P, R] | None = None, max_size: int = CLASS_MAX_SIZE, cache_holder_index: int = 0) -> Callable[P, R]:
    def _method_array_cache(func: Callable[P, R]) -> Callable[P, R]:
        cache_name = f'_method_cache_{func.__name__}_{key_cache_avoider}_'

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            obj = args[cache_holder_index]
            other_args = [a for i, a in enumerate(args) if i!=cache_holder_index]
            argument_tuple = create_function_cache_key(*other_args, **kwargs)
            if not hasattr(obj, cache_name):
                setattr(obj, cache_name, {})
            object_cache = getattr(obj, cache_name)
            if argument_tuple not in object_cache:
                if len(object_cache) >= max_size:
                    almost_all_keys = list(object_cache.keys())[0:-2]
                    uncached = [object_cache.pop(key) for key in almost_all_keys]
                result = func(*args, **kwargs)
                object_cache[argument_tuple] = {
                    'result' : result,
                    'args' : args,
                    'kwargs' : kwargs
                }
            res_collection = object_cache.pop(argument_tuple)
            object_cache[argument_tuple] = res_collection
            return res_collection['result']
            #return object_cache[argument_tuple]['result']
        
        return wrapper
    if func is not None:
        return _method_array_cache(func)
    return _method_array_cache



if __name__ == "__main__":
    key_cache_avoider = ''.join(np.random.choice([c for c in string.ascii_letters + string.digits]) for _ in range(40))

    



 #%%