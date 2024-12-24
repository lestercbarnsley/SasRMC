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


if __name__ == "__main__":
    from pathlib import Path
    from datetime import datetime

    for f in Path(__file__).parent.iterdir():
        now = datetime.now().timestamp()
        print(f.name, now - f.stat().st_mtime)
    
    


#%%