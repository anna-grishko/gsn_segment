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

# coefs = ef_coefs(outline, n_harmonics=5)
