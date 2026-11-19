import numpy as np

# Get equidistant sampled landmarks
def sample_equidist(coords, n=20):
    """
    Sample n equidistant points from the given coordinates.

    Parameters
    ----------
    coords : array
        numpy array of x,y coordinates
    n : int
        number of points to sample (including start and end points)
    
    Returns
    -------
    subset : array
        numpy array of x,y coordinates of sampled points
    """
    n = max(3, n)
    #  Calculate distance between each pair of points:
    distances = []
    for i in range(0, len(coords) - 1):
        distances.append(np.linalg.norm(coords[i + 1] - coords[i]))
    curve_length = sum(distances)
    segment_length = curve_length / (n - 1)  # n points -> (n-1) segments
    dist_cumsum = np.cumsum(distances)
    idx = []
    for j in range(0, n):
        id = np.where(dist_cumsum >= j * segment_length)
        if np.any(id):
            idx.append(id[0][0])
    # if len(coords) % (n - 1) == 0:
    #     step = int((len(coords) - 1) / (n - 1))  # shouldn't overrun coords lenght
    # else:
    #     step = int(len(coords) / (n - 1))
    # subset = coords[0:n * step:step]
    if len(idx) <= n - 1:
        idx.append(len(coords) - 1)
    subset = coords[idx]
    # subset = subset[0:n]  # make sure to return exactly n points
    subset[-1] = coords[-1]  # make sure the end point is the same as in coords

    return subset
