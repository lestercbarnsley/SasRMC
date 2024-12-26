#%%

from dataclasses import dataclass
from pytest import mark
import numpy as np

from sas_rmc.array_cache import array_cache, method_array_cache


@array_cache
def mock_func(input_arg) -> tuple:
    return (input_arg, "Result")

@mark.parametrize("input_", [
    {'a', 1},
    np.array([1]),
    "test",
    [3,1,2],
    3
])
def test_array_cache_gives_same_result(input_):

    

    res_1 = mock_func(input_)
    res_2 = mock_func(input_)
    assert res_1 == res_2
    assert id(res_1) == id(res_2)


def test_array_cache_clears():
    for i in range(51):
        mock_func(i)


@dataclass
class MockCached:

    @method_array_cache
    def mock_method(self, input_arg) -> tuple:
        return mock_func(input_arg)
    
    @method_array_cache(max_size=5)
    def mock_method_with_max(self, input_arg) -> tuple:
        return mock_func(input_arg)


def test_method_array_cache_gives_same_result():

    mock_cached = MockCached()
    res_1 = mock_cached.mock_method(5)
    res_2 = mock_cached.mock_method(5)
    assert res_1 == res_2
    assert id(res_1) == id(res_2)

def test_method_array_cache_clears():
    mock_cached = MockCached()
    
    for i in range(7):
        mock_cached.mock_method_with_max(i)




