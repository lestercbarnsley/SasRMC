import pytest
import math

import numpy as np

from sas_rmc import Vector
from sas_rmc.vector import cross

def test_vector_add():
    v = Vector(3,10,0)
    v2 = Vector(32,16,13)
    assert type(v + v2) == type(v)
    assert v + Vector.null_vector() == v

def test_vector_mag():
    v = Vector(2351, 235,1)
    v_mag = math.sqrt(2351**2 + 235**2 + 1**2)
    assert v.mag == pytest.approx(v_mag)
    assert type(v.mag) == type(v_mag)
    assert type(v.mag) != np.ndarray
    assert Vector.null_vector().mag == 0
    v_1 = Vector(1, 1, 1).unit_vector
    assert [v_1_c == pytest.approx(v_1_unit_c) for v_1_c, v_1_unit_c in zip(v_1.itercomps(), v_1.unit_vector.itercomps())]

def test_vector_to_list():
    v = Vector(2351,123,2231)
    assert len(v.to_list()) == 3
    assert type(v.to_list()) == list

def test_vector_mul():
    v = Vector(321,63.11,123)
    v2 = Vector(2311.2,31.231,231)
    assert v * v2 == pytest.approx(np.sum(np.array([321,63.11,123]) * np.array([2311.2,31.231,231])))
    assert type(v * v2) == float
    assert type(v * 321.31) == type(v)
    assert type(323.53211 * v) == type(v)
    assert 323.53211 * v == Vector(*[323.53211 * v_i for v_i in v.to_tuple()])
    assert v * -1 == -1 * v
    f = 2351.2313
    assert v * f == f * v
    assert v * -f == -f * v

def test_vector_dot():
    x_arr, y_arr = np.meshgrid(np.linspace(-51.5, +51.5, num = 101), np.linspace(-100.5, +100.5, num = 101))
    v = Vector(3.1, 2.1, 1.1)
    dot_prod = v * (x_arr, y_arr)
    assert type(dot_prod) == type(x_arr)
    assert x_arr.shape == dot_prod.shape
    dot_prod_unit_vector = v.unit_vector * (x_arr, y_arr)
    assert type(dot_prod_unit_vector) == type(x_arr)
    assert x_arr.shape == dot_prod_unit_vector.shape
    assert np.sum(dot_prod) != np.sum(dot_prod_unit_vector)

def test_vector_cross():
    v_1 = Vector(32,.123,23)
    v_2 = Vector(321.3,31,235123)
    assert type(v_1.cross(v_2)) == type(v_1)
    c = cross(v_1.to_tuple(), v_2.to_tuple())
    assert [v_c == pytest.approx(c_i) for v_c, c_i in zip((v_1.cross(v_2)).to_tuple(), c)]

def test_vector_sub():
    v = Vector(321,63.11,123)
    v2 = Vector(2311.2,31.231,231)
    assert v - v2 == v + (-1 * v2)
    assert type(v - v2) == type(v) == type(v2)
    assert v - Vector.null_vector() == v
    assert Vector.null_vector() - v == (-1 * v)

def test_vector_sub_2():
    v = Vector(32.3451,3251.23,23)
    v_2 = Vector(32.23, 3553,2351.00123)
    sc = 32.3451
    assert v - sc * v_2 == v - (sc * v_2)
    assert v - sc * v_2 == v - v_2 * sc
    assert type(v - 3325.1 * 235.123 * -23.1233 * v_2 * 312) == Vector
    v_calc_1 = 3 * v - 4 * -3 * 2 * v_2 * 12.3 
    v_calc_2 = (3* v) + (4 * 3 * 2 * 12.3 * v_2 )
    assert [v_c_1 == pytest.approx(v_c_2) for v_c_1, v_c_2 in zip(v_calc_1.to_tuple(), v_calc_2.to_tuple())]

def test_vector_div():
    v = Vector(32,3351.102,321)
    assert type(v / 32123.23) == type(v)
    assert v / 3212.5321 == (1/3212.5321 ) * v
    
def test_unit_vector():
    v = Vector(23.123,321.3251,32)
    assert type(v.unit_vector) == type(v)
    assert v.unit_vector != v
    v_1 = Vector(1,1).unit_vector
    assert v_1.mag == pytest.approx(1)
    assert [v_1_comp ==  pytest.approx(v_1_u_comp) for v_1_comp, v_1_u_comp in zip(v_1.to_tuple(), v_1.unit_vector.to_tuple())]
    assert type(v_1) == Vector
    assert Vector(0,0,0).unit_vector.mag == 0
    assert type(Vector.null_vector().unit_vector) == Vector
    assert type((v -3 * (v.unit_vector))) == Vector
    assert type(v.mag * (v.unit_vector)) == Vector

def test_vector_distance():
    v = Vector(32,5321.0,325)
    v_2 = Vector(3.23,23.123,643)
    assert type(v.distance_from_vector(v_2)) == type((v-v_2).mag)
    assert v.distance_from_vector(v) == 0

def test_rotated_basis():
    v = Vector(0.99, -0.01, -0.01)
    assert len(v.rotated_basis()) == 3





