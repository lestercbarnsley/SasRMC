from pytest import mark, approx
import numpy as np

from sas_rmc import Vector
from sas_rmc.vector import cross


@mark.parametrize(['x', 'y', 'z', 'mag'], 
                  [
                      (0, 0, 0, 0),
                      (1, 1, 0, np.sqrt(2))
                  ]
                )
def test_vector_mag(x, y, z, mag):
    assert approx(mag) == Vector(x, y, z).mag

def test_vector_iter():
    v = Vector(1,2,3)
    v.to_list()
    v.to_dict()
    v.to_numpy()
    v.to_tuple()
    assert len(v) == 3
    
@mark.parametrize(['vec_1', 'vec_2', 'vec_sum'],
                  [
                      (Vector(0,0,0), Vector(0,0,0), Vector(0,0,0)),
                      (Vector(1,2,3), Vector(4,5,6), Vector(5, 7, 9))
                  ])
def test_vector_add(vec_1: Vector, vec_2: Vector, vec_sum: Vector):
    assert vec_1 + vec_2 == vec_sum
    assert (vec_1 - vec_1).mag == 0

def test_vector_null():
    for c in Vector.null_vector():
        assert c == 0

@mark.parametrize(['vec_1', 'vec_2', 'dot_product'],
                  [
                      (Vector(0,0,0), Vector(0,0,0), 0),
                      (Vector(1,2,3), Vector(4,5,6), 32)
                  ])
def test_vector_dot(vec_1: Vector, vec_2: Vector, dot_product: float):
    assert vec_1 * vec_2 == dot_product

@mark.parametrize(['dividor', 'divided_vector'],
                  [
                      (1, Vector(10, 10, 10)),
                      (2, Vector(5, 5, 5)),
                      (5, Vector(2, 2, 2)),
                  ])
def test_vector_divide(dividor, divided_vector):
    assert Vector(10, 10, 10) / dividor == divided_vector


@mark.parametrize('vec', [
    Vector(1, 0, 0),
    Vector(1, 1, 100)
])
def test_unit_vector(vec: Vector):
    assert approx(vec.unit_vector.mag) == 1

def test_unit_vector_null():
    assert Vector.null_vector().unit_vector.mag == 0

def test_cross():
    assert isinstance(Vector(1,2,3).cross(Vector(3,4, 5)), Vector)

@mark.parametrize(
        ['vec_1', 'vec_2', 'distance'],
        [
            (Vector(0,0,0), Vector(0,0,0), 0),
            (Vector(1,1,1), Vector(1,1,1), 0),
            (Vector(3,2,1), Vector(2,2,1), 1)
        ]
)
def test_distance(vec_1: Vector, vec_2: Vector, distance: float):
    assert approx(distance) == vec_1.distance_from_vector(vec_2)

@mark.parametrize(
    'vec',
    [
        Vector(0,0,0),
        Vector(1,1,1)
    ]
)
def test_copy(vec: Vector):
    vec_copy = vec.copy()
    assert vec_copy == vec
    assert id(vec_copy) != id(vec)

@mark.parametrize(
    'vec',
    [
        Vector(0,0,0),
        Vector(1,1,1),
        Vector(5,2,1)
    ]
)
def test_project(vec: Vector):
    vec_project = vec.project_to_xy()
    assert approx(vec_project.z) == 0
    assert vec_project.x == vec.x
    assert vec_project.y == vec.y

@mark.parametrize(
    'vec',
    [
        Vector(0,0,0),
        Vector(1,1,1),
        Vector(5,2,1)
    ]
)
def test_rotated_basis(vec: Vector):
    basis_set = vec.rotated_basis()
    assert basis_set[0] == vec.unit_vector
    assert basis_set[0] * basis_set[1] == approx(0)
    assert basis_set[0] * basis_set[2] == approx(0)
    assert basis_set[1] * basis_set[2] == approx(0)

