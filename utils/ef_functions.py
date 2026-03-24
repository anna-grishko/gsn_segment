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


def ef_coefs(outline, n_harmonics, scale=False):
    n_points = outline.shape[0]

    outline_shifted = np.roll(outline[:, ], 2)
    dx = outline[:, 1] - outline_shifted[:, 1]
    dy = outline[:, 0] - outline_shifted[:, 0]
    dt = np.sqrt(np.square(dx) + np.square(dy))

    tp = np.cumsum(dt)
    tp_1 = np.append(0, tp[0:n_points - 1])
    perim_T = np.sum(dt)

    an = []
    bn = []
    cn = []
    dn = []
    for harmonic in range(n_harmonics):
        Ti = (perim_T / (2 * np.square(np.pi) * np.square(harmonic + 1)))
        numerator = 2 * (harmonic + 1) * np.pi
        an_i = Ti * np.sum((dx / dt) * (np.cos((numerator * tp) / perim_T) - np.cos((numerator * tp_1) / perim_T)))
        bn_i = Ti * np.sum((dx / dt) * (np.sin((numerator * tp) / perim_T) - np.sin((numerator * tp_1) / perim_T)))
        cn_i = Ti * np.sum((dy / dt) * (np.cos((numerator * tp) / perim_T) - np.cos((numerator * tp_1) / perim_T)))
        dn_i = Ti * np.sum((dy / dt) * (np.sin((numerator * tp) / perim_T) - np.sin((numerator * tp_1) / perim_T)))
        an = np.append(an, an_i)
        bn = np.append(bn, bn_i)
        cn = np.append(cn, cn_i)
        dn = np.append(dn, dn_i)
    return np.append(np.append(np.append(an, bn), cn), dn)


# an = np.array([-0.39940845, -0.01980334,  0.01333016, -0.02224391,  0.00092468, -0.00246493, -0.00256757])
# bn = np.array([-0.24224837,  0.03648185, -0.01691569,  0.00822352,  0.00667135, -0.00655407,  0.00596414])
# cn = np.array([-0.08655806, -0.08009734, -0.02140565,  0.00487414, -0.01000876, -0.00490515, -0.00035708])
# dn = np.array([0.28939113, -0.0316743, -0.01015062,  0.00158739, -0.00102937, 0.00123179, -0.00193434])
def iefourier(an, bn, cn, dn, n_harmonics, n_points=100, a0=0, c0=0):
    theta = np.linspace(0, 2*np.pi, n_points)
    harm_x = np.linspace(0, 0, n_points)
    harm_y = np.linspace(0, 0, n_points)
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
