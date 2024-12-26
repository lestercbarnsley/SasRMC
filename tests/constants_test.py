#%%

import numpy as np
from pytest import approx, mark

from sas_rmc.constants import np_average, np_max, np_min, np_prod, np_sum, iter_np_array

@mark.parametrize(
        ['vals', 'expected_average'],
        [
            ([1,1,1], 1),
            ([0,0,0,0], 0),
            ([1,2,3,4,5], 3),
            ([32.4, 654.61, 353.123, 541.64], 395.443)
        ]
)
def test_np_average(vals: list[float], expected_average: float):
    assert np_average(vals) == approx(expected_average)

@mark.parametrize(
        ['vals', 'expected_max'],
        [
            ([1,1,1], 1),
            ([0,0,0,0], 0),
            ([1,2,3,4,5], 5),
            ([32.4, 654.61, 353.123, 541.64], 654.61)
        ]
)
def test_np_max(vals: list[float], expected_max: float):
    assert np_max(np.array(vals)) == expected_max

@mark.parametrize(
        ['vals', 'expected_min'],
        [
            ([1,1,1], 1),
            ([0,0,0,0], 0),
            ([1,2,3,4,5], 1),
            ([32.4, 654.61, 353.123, 541.64], 32.4),
            ([-32.4, -654.61, -353.123, 541.64], -654.61)
        ]
)
def test_np_min(vals: list[float], expected_min: float):
    assert np_min(np.array(vals)) == expected_min

@mark.parametrize(
        ['vals', 'expected_prod'],
        [
            ([1,1,1], 1),
            ([0,0,0,0], 0),
            ([1,2,3,4,5], 1*2*3*4*5),
            ([32.4, 654.61, 353.123], 32.4*654.61*353.123),
            ([-1, 654.61, 2], -2 * 654.61)
        ]
)
def test_np_prod(vals: list[float], expected_prod: float):
    assert np_prod(np.array(vals)) == expected_prod

@mark.parametrize(
        ['vals', 'expected_sum'],
        [
            ([1,1,1], 3),
            ([0,0,0,0], 0),
            ([1,2,3,4,5], 15),
            ([32.4, 654.61, 353.123], 32.4+654.61+353.123),
            ([-654.61, 654.61, 2], 2)
        ]
)
def test_np_sum(vals: list[float], expected_sum: float):
    assert np_sum(np.array(vals)) == expected_sum

@mark.parametrize(
        'vals',
        [
            [1,1,1],
            [0,0,0],
            [1,2,3,4,5],
            [32.4, 654.61, 353.123, 541.64],
            [-32.4, -654.61, -353.123, 541.64]
        ]
)
def test_iter_np_array(vals: list[float]):
    for val in iter_np_array(np.array(vals)):
        assert isinstance(val, float | int)




