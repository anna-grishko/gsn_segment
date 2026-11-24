import pytest
import warnings
import numpy as np

from utils.curve_fitting_utils import sample_equidist

@pytest.mark.parametrize(
    ("num_points", "n", "expected_len", "match_msg"),
    [
        (5, 10, 5, "n=10 exceeds number of available coordinates"),
        (5, 2, 3, "n=2 is less than 3"),
        (2, 5, 2, "contains only 2 point\\(s\\)"),
    ],
)
def test_sample_equidist_warnings(num_points, n, expected_len, match_msg):
    points = np.column_stack((np.arange(num_points), np.arange(num_points)))
    with pytest.warns(UserWarning, match=match_msg):
        res = sample_equidist(points, n=n)
    assert res.shape == (expected_len, 2)


def test_sample_equidist_uneven_points():
    points = np.array([[0, 0], [0.1, 0.2], [2, 4], [3, 5], [10, 10]])
    res = sample_equidist(points, n=3)
    np.testing.assert_array_equal(res, np.array([[0, 0], [3, 5], [10, 10]]))


def test_sample_equidist_semicircle():
    theta = np.linspace(0, np.pi, 101)
    points = np.column_stack((np.cos(theta), np.sin(theta)))

    res = sample_equidist(points, n=10)
    expected = np.array([
        [ 1.00000000e+00,  0.00000000e+00],
        [ 9.40880769e-01,  3.38737920e-01],
        [ 7.70513243e-01,  6.37423990e-01],
        [ 5.09041416e-01,  8.60742027e-01],
        [ 1.87381315e-01,  9.82287251e-01],
        [-1.87381315e-01,  9.82287251e-01],
        [-5.09041416e-01,  8.60742027e-01],
        [-7.70513243e-01,  6.37423990e-01],
        [-9.40880769e-01,  3.38737920e-01],
        [-1.00000000e+00,  1.22464680e-16]
       ])
    np.testing.assert_array_almost_equal(res, expected)
    
    res_3 = sample_equidist(points, n=3)
    np.testing.assert_array_almost_equal(res_3, np.array([points[0], points[50], points[-1]]))
    np.testing.assert_array_almost_equal(res_3, np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]))

    res_all = sample_equidist(points, n=101)
    np.testing.assert_array_almost_equal(res_all, points)

