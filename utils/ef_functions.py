import warnings
import numpy as np


def ef_coefs(
        outline: np.ndarray, n_harmonics: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Calculate Elliptic Fourier coefficients for n harmonics and offset (centroid) for a given closed outline.

    Parameters
    ----------
    outline : np.ndarray
        (N, 2) array of closed outline coordinates ordered as image coordinates (row, col) / (y, x).
    n_harmonics : int
        number of harmonics.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Tuple (an, bn, cn, dn, a0, c0), where an..dn are 1D arrays of length n_harmonics,
        and (a0, c0) is the (x, y) arc-length centroid of the contour.
    """
    outline = np.asarray(outline, dtype=float)

    # Remove trailing duplicate vertex if the outline is already closed
    if len(outline) > 1 and np.allclose(outline[0], outline[-1]):
        outline = outline[:-1]

    # Remove any consecutive duplicate vertices to prevent division by zero in dx/dt and dy/dt
    diffs = np.roll(outline, -1, axis=0) - outline
    dt = np.hypot(diffs[:, 1], diffs[:, 0])
    valid = dt > 0

    start_pts = outline[valid]
    diffs = diffs[valid]
    dt = dt[valid]

    n_points = len(dt)
    if n_points == 0:
        warnings.warn(
            "Outline has fewer than 2 unique points; returning empty arrays and (0.0, 0.0) offset.",
            UserWarning,
            stacklevel=2,
        )
        empty = np.array([], dtype=float)
        return (empty, empty.copy(), empty.copy(), empty.copy(), 0.0, 0.0)
    

    x_prev = start_pts[:, 1]
    y_prev = start_pts[:, 0]
    dx = diffs[:, 1]
    dy = diffs[:, 0]

    tp = np.cumsum(dt)
    tp_prev = np.concatenate(([0.0], tp[:-1]))
    perim_T = tp[-1]

    # Kuhl & Giardina DC offset (a0 = x centroid, c0 = y centroid)
    xi = x_prev - (dx / dt) * tp_prev
    delta = y_prev - (dy / dt) * tp_prev
    a0 = float(np.sum((dx / (2.0 * dt)) * (tp**2 - tp_prev**2) + xi * dt) / perim_T)
    c0 = float(np.sum((dy / (2.0 * dt)) * (tp**2 - tp_prev**2) + delta * dt) / perim_T)

    max_harmonics = n_points // 2
    if n_harmonics > max_harmonics:
        n_harmonics = max_harmonics
        warnings.warn(
            f"'n_harmonics' must be at most half the number of unique points ({n_points}); "
            f"it has been set to {n_harmonics}.",
            UserWarning,
            stacklevel=2,
        )

    if n_harmonics < 1:
        warnings.warn(
                    f"'n_harmonics' should be >=1; returns empty arrays and preserves centroid.",
                    UserWarning,
                    stacklevel=2,
                )
        empty = np.array([], dtype=float)
        return (empty, empty.copy(), empty.copy(), empty.copy(), a0, c0)


    # Vectorize across all harmonics k = 1 .. n_harmonics -> shape (n_harmonics, 1)
    k = np.arange(1, n_harmonics + 1)[:, None]

    Ti = perim_T / (2.0 * (np.pi ** 2) * (k ** 2))
    two_pi_k_over_T = (2.0 * np.pi * k) / perim_T

    cos_diff = np.cos(two_pi_k_over_T * tp) - np.cos(two_pi_k_over_T * tp_prev)
    sin_diff = np.sin(two_pi_k_over_T * tp) - np.sin(two_pi_k_over_T * tp_prev)

    an = (Ti.T * np.sum((dx / dt) * cos_diff, axis = 1)).ravel()
    bn = (Ti.T * np.sum((dx / dt) * sin_diff, axis = 1)).ravel()
    cn = (Ti.T * np.sum((dy / dt) * cos_diff, axis = 1)).ravel()
    dn = (Ti.T * np.sum((dy / dt) * sin_diff, axis = 1)).ravel()

    return (an, bn, cn, dn, a0, c0)


def iefourier(
        an: np.ndarray, 
        bn: np.ndarray, 
        cn: np.ndarray, 
        dn: np.ndarray, 
        n_harmonics: int | None = None, 
        n_points: int = 100, 
        a0: float = 0.0, 
        c0: float = 0.0
) -> np.ndarray:
    """
    Perform inverse elliptic Fourier transformation to reconstruct a closed 2D contour.

    Parameters
    ----------
    an, bn, cn, dn : np.ndarray
        1D arrays of Elliptic Fourier coefficients.
    n_harmonics : int, optional
        Numebr of harmonics to use to reconstruc the outline.
    n_points : int, default=100
        Number of x,y coordinates on the reconstructed outline.
    a0, c0 : float, default=0.0
        Centroid coordinates (x, y) of the contour.

    Returns
    -------
    np.ndarray
        (n_points, 2) array of reconstructed coordinates ordered as (row, col) / (y, x).

    """
    an = np.asarray(an, dtype=float)
    bn = np.asarray(bn, dtype=float)
    cn = np.asarray(cn, dtype=float)
    dn = np.asarray(dn, dtype=float)

    max_harmonics = len(an)
    if n_harmonics is None:
        n_harmonics = max_harmonics
    elif n_harmonics < 1:
        warnings.warn(
            f"Requested n_harmonics={n_harmonics} is less  than 1; returning centroid coordinates.",
            UserWarning,
            stacklevel=2,
        )
        return np.column_stack((np.full(n_points, c0), np.full(n_points, a0)))
    elif max_harmonics == 0:
        warnings.warn(
            f"Empty coefficients; returning centroid coordinates.",
            UserWarning,
            stacklevel=2,
        )
        return np.column_stack((np.full(n_points, c0), np.full(n_points, a0)))
    elif n_harmonics > max_harmonics:
        warnings.warn(
            f"Requested n_harmonics={n_harmonics} exceeds available coefficients ({max_harmonics}); "
            f"using n_harmonics={max_harmonics}.",
            UserWarning,
            stacklevel=2,
        )
        n_harmonics = max_harmonics
    
    theta = np.linspace(0, 2.0 * np.pi, n_points, endpoint=False)
    k = np.arange(1, n_harmonics + 1)[:, None]
    angles = k * theta  # shape (n_harmonics, n_points)

    cos_k = np.cos(angles)
    sin_k = np.sin(angles)

    x = a0 + an[:n_harmonics] @ cos_k + bn[:n_harmonics] @ sin_k
    y = c0 + cn[:n_harmonics] @ cos_k + dn[:n_harmonics] @ sin_k
 
    return np.column_stack((y, x))
