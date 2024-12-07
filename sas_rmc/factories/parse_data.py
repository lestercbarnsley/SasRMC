#%%
import inspect

import pandas as pd

  
truth_dict = {
    'ON' : True,
    'OFF' : False,
    'on' : True,
    'off' : False,
    'On': True,
    'Off': False,
    'True' :  True,
    'TRUE' : True,
    'true' : True,
    'False' :  False,
    'FALSE' : False,
    'false' : False
} # I'm sure I haven't come close to fully covering all the wild and creative ways users could say "True" or "False"

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

