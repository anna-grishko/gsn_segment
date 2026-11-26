import pytest
import warnings
import numpy as np

from utils.ef_functions import ef_coefs, iefourier


### Elliptic Fourier ###
def test_ef_coefs_unit_circle():
    """For a unit circle x = cos(t), y = sin(t),
    first harmonic has A_1=1, D_1=1, all others harmonics ~0 and (a0, c0)=(0, 0)."""
    theta = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    # Image coordinates: (row, col) -> (y, x)
    outline = np.column_stack((np.sin(theta), np.cos(theta)))

    an, bn, cn, dn, a0, c0 = ef_coefs(outline, n_harmonics=4)

    assert np.isclose(a0, 0.0)
    assert np.isclose(c0, 0.0)

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


def test_ef_coefs_ellipse():
    """For an axis-aligned ellipse with semi-axes rx=5, ry=2, A_1=4.455 and D_1=2.178"""
    theta = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    rx, ry = 5.0, 2.0
    outline = np.column_stack((ry * np.sin(theta), rx * np.cos(theta)))

    an, bn, cn, dn, a0, c0 = ef_coefs(outline, n_harmonics=20)

    assert np.isclose(a0, 0.0)
    assert np.isclose(c0, 0.0)

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


def test_ef_coefs_rotated_ellipse():
    """A 45-degree rotated ellipse exercises non-zero an, bn, cn, and dn together."""
    n_points = 300
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    rx, ry = 4.0, 1.5
    phi = np.pi / 4  # 45 degrees

    x = rx * np.cos(theta) * np.cos(phi) - ry * np.sin(theta) * np.sin(phi)
    y = rx * np.cos(theta) * np.sin(phi) + ry * np.sin(theta) * np.cos(phi)
    rotated = np.column_stack((y, x))

    an, bn, cn, dn, a0, c0 = ef_coefs(rotated, n_harmonics=15)
    assert not np.allclose(an, 0.0, atol=1e-2)
    assert not np.allclose(bn, 0.0, atol=1e-2)
    assert not np.allclose(cn, 0.0, atol=1e-2)
    assert not np.allclose(dn, 0.0, atol=1e-2)
    assert np.isclose(a0, 0.0, atol=1e-6)
    assert np.isclose(c0, 0.0, atol=1e-6)

    # Harmonic power A_k^2 + B_k^2 + C_k^2 + D_k^2 is invariant to rotation
    unrotated = np.column_stack((ry * np.sin(theta), rx * np.cos(theta)))
    an0, bn0, cn0, dn0, _, _ = ef_coefs(unrotated, n_harmonics=15)
    power_rot = an**2 + bn**2 + cn**2 + dn**2
    power_unrot = an0**2 + bn0**2 + cn0**2 + dn0**2
    np.testing.assert_allclose(power_rot, power_unrot, atol=1e-4)


def test_ef_coefs_translation_invariance():
    """Shifting the outline anywhere in the image does not change EF coefficients."""
    theta = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    outline = np.column_stack((3.0 * np.sin(theta), 2.0 * np.cos(theta)))

    shift_y, shift_x = 10.0, 20.0
    shifted_outline = outline + np.array([shift_y, shift_x])

    an1, bn1, cn1, dn1, a0_1, c0_1 = ef_coefs(outline, n_harmonics=5)
    an2, bn2, cn2, dn2, a0_2, c0_2 = ef_coefs(shifted_outline, n_harmonics=5)

    assert np.isclose(a0_2, shift_x, atol=1e-12)
    assert np.isclose(c0_2, shift_y, atol=1e-12)
    np.testing.assert_allclose((an1, bn1, cn1, dn1), (an2, bn2, cn2, dn2), atol=1e-12)


def test_ef_coefs_preclosed_and_duplicate_vertices():
    """Pre-closed contours and repeated consecutive vertices do not alter results or produce NaNs."""
    theta = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    outline = np.column_stack((np.sin(theta), np.cos(theta)))

    closed_with_dups = np.vstack([outline[:10], outline[9:10], outline[10:], outline[0:1]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        coefs_clean = ef_coefs(outline, n_harmonics=5)
        coefs_dirty = ef_coefs(closed_with_dups, n_harmonics=5)

    np.testing.assert_allclose(coefs_clean[:4], coefs_dirty[:4], atol=1e-12)
    np.testing.assert_allclose(coefs_clean[4:], coefs_dirty[4:], atol=1e-12)


def test_ef_coefs_nyquist_warning_and_clamping():
    """Requesting n_harmonics > n_points // 2 emits a UserWarning and clamps to n_points // 2."""
    theta = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    outline = np.column_stack((np.sin(theta), np.cos(theta)))

    with pytest.warns(UserWarning, match="must be at most half the number of unique points"):
        an, bn, cn, dn, a0, c0 = ef_coefs(outline, n_harmonics=8)

    # Clamped to 10 // 2 = 5 harmonics per coefficient array
    for arr in (an, bn, cn, dn):
        assert arr.shape == (5,)

    assert np.isclose(a0, 0.0)
    assert np.isclose(c0, 0.0)


def test_ef_coefs_degenerate_outline():
    """An outline with fewer than 2 unique vertices warns and returns 4 empty arrays."""
    single_point = np.array([[5.0, 5.0], [5.0, 5.0]])

    with pytest.warns(UserWarning, match="Outline has fewer than 2 unique points; returning empty arrays and \\(0.0, 0.0\\) offset."):
        an, bn, cn, dn, a0, c0 = ef_coefs(single_point, n_harmonics=3)
    
    for arr in (an, bn, cn, dn):
        assert arr.size == 0
    assert (a0, c0) == (0.0, 0.0)


def test_ef_coefs_invalid_n_harmonics_warning():
    """Requesting n_harmonics < 1 on a valid outline warns, returns empty arrays, and preserves centroid."""
    theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    outline = np.column_stack((np.sin(theta) + 10.0, np.cos(theta) - 4.0))

    with pytest.warns(UserWarning, match="'n_harmonics' should be >=1; returns empty arrays and preserves centroid."):
        an, bn, cn, dn, a0, c0 = ef_coefs(outline, n_harmonics=0)

    for arr in (an, bn, cn, dn):
        assert arr.size == 0
    assert np.isclose(a0, -4.0)
    assert np.isclose(c0, 10.0)

    """Requesting n_harmonics > n_points // 2 on a valid outline warns, set n_harmonics to n_points // 2."""
    with pytest.warns(UserWarning, match="'n_harmonics' must be at most half the number of unique points \\(20\\); it has been set to 10."):
        an, bn, cn, dn, a0, c0 = ef_coefs(outline, n_harmonics=15)
    for arr in (an, bn, cn, dn):
        assert arr.size == 10
    assert np.isclose(a0, -4.0)
    assert np.isclose(c0, 10.0)



### Inverse Elliptic Fourier ###
def test_iefourier_reconstruct_circle():
    """Forward EFA followed by inverse EFA accurately reconstructs translated circle."""
    n_points = 300
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    cy, cx = 45.0, 120.0

    # Shifted circle: exact reconstruction with 1 harmonic
    circle = np.column_stack((np.sin(theta) + cy, np.cos(theta) + cx))
    an = np.array([1.0])
    bn = np.array([0.0])
    cn = np.array([0.0])
    dn = np.array([1.0])
    rec_circle = iefourier(an, bn, cn, dn, n_harmonics=1, n_points=n_points, a0=cx, c0=cy)
    np.testing.assert_allclose(rec_circle, circle, atol=1e-12)


def test_iefourier_reconstruct_ellipse():
    """Forward EFA followed by inverse EFA accurately reconstructs translated ellipse."""
    n_points = 300
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    cy, cx = 45.0, 120.0

    # Shifted ellipse: accurate reconstruction with 15 harmonics
    ellipse = np.column_stack((2.0 * np.sin(theta) + cy, 5.0 * np.cos(theta) + cx))
    an, bn, cn, dn, a0, c0 = ef_coefs(ellipse, n_harmonics=15)
    rec_ellipse = iefourier(an, bn, cn, dn, n_points=n_points, a0=a0, c0=c0)

    # Check bounding box and centroid match closely (theta spacing differs slightly from arc-length t)
    assert np.isclose(rec_ellipse[:, 0].min(), cy - 2.0, atol=1e-2)
    assert np.isclose(rec_ellipse[:, 0].max(), cy + 2.0, atol=1e-2)
    assert np.isclose(rec_ellipse[:, 1].min(), cx - 5.0, atol=1e-2)
    assert np.isclose(rec_ellipse[:, 1].max(), cx + 5.0, atol=1e-2)
    assert np.isclose(rec_ellipse[:, 0].mean(), cy, atol=1e-2)
    assert np.isclose(rec_ellipse[:, 1].mean(), cx, atol=1e-2)


def test_iefourier_warns():
    an = np.array([1.0, 0.0])
    bn = np.array([0.0, 0.0])
    cn = np.array([0.0, 0.0])
    dn = np.array([1.0, 0.0])

    # Requesting more harmonics than provided in an..dn warns and uses all available harmonics.
    with pytest.warns(UserWarning, match="Requested n_harmonics=5 exceeds available coefficients \\(2\\); using n_harmonics=2."):
        rec = iefourier(an, bn, cn, dn, n_harmonics=5, n_points=50)

    assert rec.shape == (50, 2)

    # Requesting more zero harmonics warns and returns centroid coordinates.
    with pytest.warns(UserWarning, match="Requested n_harmonics=0 is less  than 1; returning centroid coordinates."):
        rec = iefourier(an, bn, cn, dn, n_harmonics=0, n_points=50, a0=3.5, c0=12.5)

    assert rec.shape == (50, 2)
    np.testing.assert_allclose(rec, np.tile([12.5, 3.5], (50, 1)))


    # Supplying empty coefficients warns and returns centroid coordinates.
    an = np.array([])
    bn = np.array([])
    cn = np.array([])
    dn = np.array([])
    with pytest.warns(UserWarning, match="Empty coefficients; returning centroid coordinates."):
        rec = iefourier(an, bn, cn, dn, n_harmonics=5, n_points=50)
    
    assert rec.shape == (50, 2)


def test_iefourier_rotated_ellipse():
    """A 45-degree rotated ellipse exercises non-zero an, bn, cn, and dn together."""
    n_points = 300
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    rx, ry = 4.0, 1.5
    phi = np.pi / 4  # 45 degrees

    x = rx * np.cos(theta) * np.cos(phi) - ry * np.sin(theta) * np.sin(phi)
    y = rx * np.cos(theta) * np.sin(phi) + ry * np.sin(theta) * np.cos(phi)
    rotated = np.column_stack((y, x))

    an, bn, cn, dn, a0, c0 = ef_coefs(rotated, n_harmonics=15)
    rec = iefourier(an, bn, cn, dn, n_points=n_points, a0=a0, c0=c0)
    # Rotate by phi angle
    diag_major = (rec[:, 1] + rec[:, 0]) / np.sqrt(2)  # (x*cosphi+y*sinphi)
    diag_minor = (-rec[:, 1] + rec[:, 0]) / np.sqrt(2)  # (-x*sinphi+y*cosphi)
    # Check major/minor extents along the 45-degree diagonals
    assert np.isclose(diag_major.max(), rx, atol=1e-2)
    assert np.isclose(diag_minor.max(), ry, atol=1e-2)
