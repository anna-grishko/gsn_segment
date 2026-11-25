import pytest
import warnings
import numpy as np

from utils.ef_functions import ef_coefs, iefourier


def test_ef_coefs_unit_circle():
    """For a unit circle x = cos(t), y = sin(t), first harmonic has A_1=1, D_1=1, and all others ~0."""
    n_points = 1000
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    # Image coordinates: (row, col) -> (y, x)
    outline = np.column_stack((np.sin(theta), np.cos(theta)))

    an, bn, cn, dn = ef_coefs(outline, n_harmonics=4)

    # First harmonic should match the circle's semi-axes (rx=1, ry=1)
    assert np.isclose(an[0], 1.0, atol=1e-2)
    assert np.isclose(bn[0], 0.0, atol=1e-2)
    assert np.isclose(cn[0], 0.0, atol=1e-2)
    assert np.isclose(dn[0], 1.0, atol=1e-2)

    # Higher harmonics for a pure circle should vanish
    np.testing.assert_allclose(an[1:], 0.0, atol=1e-3)
    np.testing.assert_allclose(bn[1:], 0.0, atol=1e-3)
    np.testing.assert_allclose(cn[1:], 0.0, atol=1e-3)
    np.testing.assert_allclose(dn[1:], 0.0, atol=1e-3)


def test_ef_coefs_ellipse_axes():
    """For an axis-aligned ellipse with semi-axes rx=5, ry=2, A_1=5 and D_1=2."""
    n_points = 1000
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    rx, ry = 5.0, 2.0
    outline = np.column_stack((ry * np.sin(theta), rx * np.cos(theta)))

    an, bn, cn, dn = ef_coefs(outline, n_harmonics=20)

    # Exact first-harmonic arc-length EFA values for rx=5, ry=2
    assert np.isclose(an[0], 4.4551998, atol=1e-4)
    assert np.isclose(dn[0], 2.1778912, atol=1e-4)

    # Symmetry: no sine terms for x (bn=0), no cosine terms for y (cn=0), and zero even harmonics
    np.testing.assert_allclose(an[1::2], 0.0, atol=1e-5)
    np.testing.assert_allclose(dn[1::2], 0.0, atol=1e-5)


