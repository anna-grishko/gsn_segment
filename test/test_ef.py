import pytest
import warnings
import numpy as np

from utils.ef_functions import ef_coefs, iefourier


def test_ef_coefs_unit_circle():
    """For a unit circle x = cos(t), y = sin(t), first harmonic has A_1=1, D_1=1, and all others ~0."""
    theta = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    # Image coordinates: (row, col) -> (y, x)
    outline = np.column_stack((np.sin(theta), np.cos(theta)))

    an, bn, cn, dn = ef_coefs(outline, n_harmonics=4)

    # First harmonic should match the circle's semi-axes (rx=1, ry=1)
    assert np.isclose(an[0], 1.0, atol=1e-5)
    assert np.isclose(bn[0], 0.0, atol=1e-5)
    assert np.isclose(cn[0], 0.0, atol=1e-5)
    assert np.isclose(dn[0], 1.0, atol=1e-5)

    # Higher harmonics for a pure circle should vanish
    np.testing.assert_allclose(an[1:], 0.0, atol=1e-5)
    np.testing.assert_allclose(bn[1:], 0.0, atol=1e-5)
    np.testing.assert_allclose(cn[1:], 0.0, atol=1e-5)
    np.testing.assert_allclose(dn[1:], 0.0, atol=1e-5)


def test_ef_coefs_ellipse_axes():
    """For an axis-aligned ellipse with semi-axes rx=5, ry=2, A_1=5 and D_1=2."""
    theta = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    rx, ry = 5.0, 2.0
    outline = np.column_stack((ry * np.sin(theta), rx * np.cos(theta)))

    an, bn, cn, dn = ef_coefs(outline, n_harmonics=20)

    # Exact first-harmonic EFA values
    assert np.isclose(an[0], 4.4552260, atol=1e-5)
    assert np.isclose(dn[0], 2.1779040, atol=1e-5)

    # Zero even harmonics for an, dn
    np.testing.assert_allclose(an[1::2], 0.0, atol=1e-5)
    np.testing.assert_allclose(dn[1::2], 0.0, atol=1e-5)

    # Zero harmonics for bn, cn
    np.testing.assert_allclose(bn, 0.0, atol=1e-5)
    np.testing.assert_allclose(cn, 0.0, atol=1e-5)

    # Reconstructing at t=0 (sum of an) and t=T/4 (D1 - D3 + D5 - D7...) recovers rx and ry
    assert np.isclose(np.sum(an), rx, atol=1e-3)
    assert np.isclose(np.sum(dn[0::4])-np.sum(dn[2::4]), ry, atol=1e-3)


def test_ef_coefs_translation_invariance():
    """Shifting the outline anywhere in the image does not change EF coefficients."""
    theta = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    outline = np.column_stack((3.0 * np.sin(theta), 2.0 * np.cos(theta)))
    shifted_outline = outline + np.array([10.0, 20.0])

    coefs_orig = ef_coefs(outline, n_harmonics=5)
    coefs_shifted = ef_coefs(shifted_outline, n_harmonics=5)

    np.testing.assert_array_almost_equal(coefs_orig, coefs_shifted)


def test_ef_coefs_preclosed_and_duplicate_vertices():
    """Pre-closed contours and repeated consecutive vertices do not alter results or produce NaNs."""
    theta = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    outline = np.column_stack((np.sin(theta), np.cos(theta)))

    closed_with_dups = np.vstack([outline[:10], outline[9:10], outline[10:], outline[0:1]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        coefs_clean = ef_coefs(outline, n_harmonics=5)
        coefs_dirty = ef_coefs(closed_with_dups, n_harmonics=5)

    for clean_arr, dirty_arr in zip(coefs_clean, coefs_dirty):
        assert not np.any(np.isnan(dirty_arr))
        np.testing.assert_allclose(clean_arr, dirty_arr, atol=1e-12)
    np.testing.assert_array_almost_equal(coefs_clean, coefs_dirty)


def test_ef_coefs_nyquist_warning_and_clamping():
    """Requesting n_harmonics > n_points // 2 emits a UserWarning and clamps to n_points // 2."""
    theta = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    outline = np.column_stack((np.sin(theta), np.cos(theta)))

    with pytest.warns(UserWarning, match="must be at most half the number of unique points"):
        an, bn, cn, dn = ef_coefs(outline, n_harmonics=8)

    # Clamped to 10 // 2 = 5 harmonics per coefficient array
    for arr in (an, bn, cn, dn):
        assert arr.shape == (5,)


def test_ef_coefs_degenerate_outline():
    """An outline with fewer than 2 unique vertices warns and returns 4 empty arrays."""
    single_point = np.array([[5.0, 5.0], [5.0, 5.0]])

    with pytest.warns(UserWarning) as record:
        an, bn, cn, dn = ef_coefs(single_point, n_harmonics=3)

    assert len(record) == 2
    assert "must be at most half the number of unique points" in str(record[0].message)
    assert "'n_harmonics' should be >=1" in str(record[1].message)
    
    for arr in (an, bn, cn, dn):
        assert arr.size == 0