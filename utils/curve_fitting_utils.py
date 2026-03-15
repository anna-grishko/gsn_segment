import numpy as np

def cart2pol(x, y):
    '''
    :param x, y: carthesian coordinates (x, y)
    :return: polar coordinates (rho, phi)
    '''
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi, x_shift=0):
    '''
    :param rho, phi: polar coordinates (rho, phi)
    :return: carthesian coordinates (x, y)
    '''
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    x = x + (x_shift - x[0])
    return (x, y)


def orig_to_0_0(coords):
    '''
    :param coords: x,y coords, np.array
    :return: shifted set of coords [x_new, y_new] so (x_0,y_0) = (0,0)
    '''
    return coords - coords[0]


def orig_to_mid_AB(coords):
    '''
    :param coords: x,y coords, np.array
    A = first point, B = last point
    :return: shifted set of coords [x_new, y_new]
    '''
    new_orig = (coords[0] + coords[-1]) / 2
    new_coords = coords - new_orig
    new_coords[0, 1] = 0
    return new_coords



# Restore n harmonics from FFT
def y_restore(x_sample, fft_y, harmonics):
    a0 = fft_y[0].real / len(x_sample)
    res = a0
    for n in range(1, harmonics + 1):
        a = 2 * fft_y[n].real / len(x_sample)
        b = -2 * fft_y[n].imag / len(x_sample)
        res += a * np.cos(2 * n * x_sample) + \
               b * np.sin(2 * n * x_sample)
    return res


# Get equidistant sampled landmarks
def sample_equidist(coords, n=20):
    '''
    :param coords:
    :param n: n points to sample (including start and end points)
    :return:
    '''
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
