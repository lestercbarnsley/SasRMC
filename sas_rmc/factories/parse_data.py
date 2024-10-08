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

def parse_value_frame(value_frame: pd.DataFrame) -> dict:
    d = {}
    for _, row in value_frame.iterrows():
        param_name = row.iloc[0]
        param_value = row.iloc[1]
        if any(not p.strip() for p in (param_name, param_value)):
            continue
        if any('#' in p.strip() for p in (param_name, param_value)):
            continue
        v = param_value.strip()
        if v.lower() in truth_dict:
            v = truth_dict[v.lower()]
        d[param_name.strip()] = v
    return d


