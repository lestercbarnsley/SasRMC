#%%

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

def truth_function(truth: str) -> bool: # raises ValueError
    for truthy in ['On', 'True', 'Yes']:
        if truth.lower() == truthy.lower():
            return True
    for falsy in ['Off', 'False', 'No']:
        if truth.lower == falsy.lower():
            return False
    raise ValueError("this is neither truthy nor falsy")


def is_bool_in_truth_dict(s: str) -> bool:
    return s in truth_dict     

def _is_numeric_type(s, t = int):
    try:
        v = t(s)
        return True
    except ValueError:
        return False

def is_int(s: str) -> bool:
    return _is_numeric_type(s, t = int)

def is_float(s: str) -> bool:
    return _is_numeric_type(s, t = float)

def add_row_to_dict(d: dict, param_name: str, param_value: str) -> None:
    if not param_name.strip():
        return
    if r'#' in param_name.strip():
        return
    v = param_value.strip()
    if not v:
        return
    for is_f, t in [
            (is_int, int),
            (is_float, float),
            (is_bool_in_truth_dict, lambda v_ : truth_dict[v_]),
            (lambda _ : True, lambda v_ : v_)]:
        if is_f(v):
            d[param_name] = t(v)
            return

def dataframe_to_config_dict(dataframe: pd.DataFrame) -> dict:
    config_dict = dict()
    for _, row in dataframe.iterrows():
        param_name = row.iloc[0]
        param_value = row.iloc[1]
        add_row_to_dict(config_dict, param_name, param_value)
    return config_dict

def dataseries_to_config_dict(dataseries: pd.Series) -> dict:
    config_dict = dict()
    for h, v in dataseries.items():
        add_row_to_dict(config_dict, h, v)
    return config_dict

