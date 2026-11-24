import warnings
import numpy as np

def sample_equidist(coords: np.ndarray, n=20) -> np.ndarray:
    """
    Sample n equidistant points from the given coordinates of an open contour.

    Parameters
    ----------
    coords : np.ndarray
        numpy array of x,y coordinates
    n : int
        number of points to sample (including start and end points)
    
    Returns
    -------
    subset : np.ndarray
        numpy array of x,y coordinates of sampled points
    """

    if len(coords) <= 3:
        warnings.warn(
            f"`coords` contains only {len(coords)} point(s); returning a copy as-is.",
            UserWarning,
            stacklevel=2,
        )
        return coords.copy()

    if n < 3:
        warnings.warn(
            f"Requested n={n} is less than 3; clamping n to 3.",
            UserWarning,
            stacklevel=2,
        )
        n = 3  # sample at least 3 points
    elif n > len(coords):
        warnings.warn(
            f"Requested n={n} exceeds number of available coordinates ({len(coords)}); returning copy as-is.",
            UserWarning,
            stacklevel=2,
        )
        return coords.copy()

    #  Calculate Euclidean distance between consecutive points
    diffs = np.diff(coords, axis=0)
    segment_lengths = np.hypot(diffs[:, 0], diffs[:, 1])

    cum_dist = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    target_dists = np.linspace(0.0, cum_dist[-1], n)

    # Indices of the vertices immediately after (right) and before (left) each target
    right_idx = np.clip(np.searchsorted(cum_dist, target_dists), 1, len(coords) - 1)
    left_idx = right_idx - 1

    choose_left = (target_dists - cum_dist[left_idx]) < (cum_dist[right_idx] - target_dists)
    idx = np.where(choose_left, left_idx, right_idx)
    
    idx[-1] = len(coords) - 1  # make sure the end point is the same as in coords

    return coords[idx]
