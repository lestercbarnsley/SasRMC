#%%
from typing import Callable

import numpy as np

from sas_rmc.array_cache import array_cache

@array_cache
def functional_cache(arr: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    def adder(arr_2: np.ndarray):
        #print(id(arr))
        return arr + arr_2
    return adder

add_to_threes = functional_cache(np.array([3,3,3,3]))

for _ in range(10):
    add_to_threes(np.random.rand(4))

