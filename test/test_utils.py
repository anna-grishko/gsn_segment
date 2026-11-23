import pytest
import numpy as np

from utils.curve_fitting_utils import sample_equidist


def test_sampling():
    points = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
    res = sample_equidist(points, n=10)
    assert res.shape[0] == 5

    res = sample_equidist(points, n=4)
    assert res.shape[0] == 4

    res = sample_equidist(points, n=2)
    assert res.shape[0] == 3

    res = sample_equidist(points, n=5)
    np.testing.assert_array_equal(res, points)

    points = np.array([[0, 0], [0.1, 0.2], [2, 4], [3, 5], [10, 10]])
    res = sample_equidist(points, n=3)
    np.testing.assert_array_equal(res, np.array([[ 0,  0], [ 3,  5], [10, 10]]))

    theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
    points = np.column_stack((np.cos(theta), np.sin(theta)))
    res = sample_equidist(points, n=10)
    np.testing.assert_array_almost_equal(
        res,
        np.array(
            [[ 1.        ,  0.        ],
            [ 0.77051324,  0.63742399],
            [ 0.18738131,  0.98228725],
            [-0.48175367,  0.87630668],
            [-0.92977649,  0.36812455],
            [-0.95105652, -0.30901699],
            [-0.53582679, -0.84432793],
            [ 0.12533323, -0.9921147 ],
            [ 0.72896863, -0.68454711],
            [ 0.99802673, -0.06279052]])
    )
