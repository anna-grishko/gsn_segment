import warnings
import numpy as np

# folder = os.path.join('..', 'data', 'segmented_curves', 'final', 'Maxwell')
# os.listdir(folder)
# file_idx = 1  # take the first one
# file_path = os.path.join(folder, os.listdir(folder)[file_idx], 'gsn_curve_registered.csv')
# coordinates = pd.read_csv(file_path, header=None)
# coordinates = coordinates.rename(columns={0: "y", 1: "x"})
# # close the outline
# new_dat = pd.DataFrame(np.array([np.zeros(coordinates.shape[0]), coordinates.loc[:, "x"]]).T, columns=['y', 'x'])
# outline = pd.concat([coordinates, new_dat], axis=0, ignore_index=True)


def ef_coefs(outline: np.ndarray, n_harmonics: int) -> np.ndarray:
    """
    Calculate Elliptic Fourier coefficients for n harmonics for a given closed outline.

    Parameters
    ----------
    outline : np.ndarray
        (N, 2) array of closed outline coordinates ordered as image coordinates (row, col) / (y, x).
    n_harmonics : int
        number of harmonics.
    
    Returns
    -------
    np.ndarray
        1D array of EF coefficients appended in this order: [A_1..A_n, B_1..B_n, C_1..C_n, D_1..D_n].
    """
    outline = np.asarray(outline, dtype=float)

    # Remove trailing duplicate vertex if the outline is already closed
    if len(outline) > 1 and np.allclose(outline[0], outline[-1]):
        outline = outline[:-1]

    # Remove any consecutive duplicate vertices to prevent division by zero in dx/dt
    diffs = outline - np.roll(outline, 1, axis=0)
    dx = diffs[:, 1]
    dy = diffs[:, 0]
    dt = np.hypot(dx, dy)
    valid = dt > 0
    diffs = diffs[valid]
    dt = dt[valid]

    n_points = len(dt)
    max_harmonics = n_points // 2

    if n_harmonics > max_harmonics:
        n_harmonics = max_harmonics
        warnings.warn(
            f"'n_harmonics' must be at most half the number of points ({n_points}); "
            f"it has been set to {n_harmonics}.",
            UserWarning,
            stacklevel=2,
        )

    if n_harmonics < 1:
        return np.array([], dtype=float)

    tp = np.cumsum(dt)
    tp_prev = np.concatenate(([0.0], tp[:-1]))
    perim_T = tp[-1]

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

    return (an, bn, cn, dn)


# an = np.array([-0.39940845, -0.01980334,  0.01333016, -0.02224391,  0.00092468, -0.00246493, -0.00256757])
# bn = np.array([-0.24224837,  0.03648185, -0.01691569,  0.00822352,  0.00667135, -0.00655407,  0.00596414])
# cn = np.array([-0.08655806, -0.08009734, -0.02140565,  0.00487414, -0.01000876, -0.00490515, -0.00035708])
# dn = np.array([0.28939113, -0.0316743, -0.01015062,  0.00158739, -0.00102937, 0.00123179, -0.00193434])
def iefourier(an, bn, cn, dn, n_harmonics, n_points=100, a0=0, c0=0):
    """
    Perform inverse elliptic Fourier transformation, e.g.
    reconstructs a closed 2D contour from its harmonic EF coefficients.

    Parameters
    ----------
    an : ndarray
        A elliptic Fourier coefficients.
    bn : ndarray
        B elliptic Fourier coefficients.
    cn : ndarray
        C elliptic Fourier coefficients.
    dn : ndarray
        D elliptic Fourier coefficients.
    n_harmonics : int
        numebr of harmonics to use to reconstruc the outline.
    n_points : int
        number of x,y coordinates on the reconstructed outline.
    a0 : float
        x coordinate of the  centroid of the configuration.
    c0 : float
        y coordinate of the centroid of the configuration.

    Returns
    -------
    x : ndarray
        x coordinates of the contour reconstruction.
    y : ndarray
        y coordinates of the contour reconstruction.

    """

    theta = np.linspace(0, 2*np.pi, n_points)
    harm_x = np.zeros(n_points)
    harm_y = np.zeros(n_points)
    n_harmonics = n_harmonics if an.__len__() > n_harmonics else an.__len__()
    for harmonic in range(n_harmonics):
        harm_x_i = an[harmonic] * np.cos((harmonic+1) * theta) + bn[harmonic] * np.sin((harmonic+1) * theta)
        harm_y_i = cn[harmonic] * np.cos((harmonic+1) * theta) + dn[harmonic] * np.sin((harmonic+1) * theta)
        harm_x = np.c_[harm_x, harm_x_i]
        harm_y = np.c_[harm_y, harm_y_i]
    x = a0/2 + np.sum(harm_x, axis=1)
    y = c0/2 + np.sum(harm_y, axis=1)

    return {
        "x": x,
        "y": y
    }
