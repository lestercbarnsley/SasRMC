#%%
import inspect

import pandas as pd


def parse_value_frame(value_frame: pd.DataFrame) -> dict:
    d = {}
    for _, row in value_frame.iterrows():
        param_name = row.iloc[0]
        param_value = row.iloc[1]
        if any(not p.strip() for p in (param_name, param_value)):
            continue
        if any('#' in p.strip() for p in (param_name, param_value)):
            continue
        d[param_name.strip()] = param_value.strip()
    return d


def coerce_types(func):

        def wrapper(*args, **kwargs):

            coerced_kwargs = {}
            no_kwargs_only = all(v.kind != inspect._ParameterKind.KEYWORD_ONLY for v in inspect.signature(func).parameters.values())
            for k, v in inspect.signature(func).parameters.items():
                if k in kwargs:
                    if v.kind == inspect._ParameterKind.KEYWORD_ONLY or no_kwargs_only:
                        coerced_kwargs[k] = v.annotation(kwargs[k])
                    else:
                        coerced_kwargs[k] = kwargs[k]
            return func(*args, **coerced_kwargs)
        return wrapper

if __name__ == "__main__":

    def validate_bool(s: str | float | int) -> bool:
        if s == 1:
            return True
        if s == 0:
            return False
        if isinstance(s, int | float):
            return bool(s)
        if s.lower() == "true":
            return True
        if s.lower() == 'false':
            return False
        if s.lower() == "on":
            return True
        if s.lower() == "off":
            return False
        if s.lower() == 'y' or 'yes' in s.lower():
            return True
        if s.lower() == 'n' or 'no' in s.lower():
            return False
        return bool(s)

    from dataclasses import dataclass

    from functools import wraps

    from collections.abc import Callable

    from typing import ParamSpec, TypeVar

    T = TypeVar("T")

    def validated_dataclass(cls: type[T]) -> type[T]:
        dclas = dataclass(cls)
        print(dclas.__name__)
                



        @dataclass
        class ValidatedClass(dclas):

            def __setattr__(self, name: str, value) -> None:
                field = self.__dataclass_fields__.get(name)
                if field is not None:
                    t = field.type
                    if isinstance(value, t):
                        return super().__setattr__(name, value)
                    
                    try:
                        if t == bool:
                            return super().__setattr__(name, validate_bool(value))
                        return super().__setattr__(name, t(value))
                    except Exception:
                        raise
        
        ValidatedClass.__name__ = dclas.__name__
        ValidatedClass.__qualname__ = dclas.__qualname__
        return ValidatedClass
    
   

    @validated_dataclass
    class Test:
        x: float
        y: float
        t: bool

    t = Test(3, 4, t ='yes')
    print(t)
    print(t.t)


    
    


#%%