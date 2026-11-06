import numpy as np
import pandas as pd
import scipy
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

def draw_conf_ellipse(x, y, ax, color, conf=0.95, size=100, plot_center=True, linestyle='solid'):
    z_quant = scipy.stats.norm.ppf(conf + (1 - conf) / 2, loc=0, scale=1)
    cov = np.cov(x, y)
    val, rot = np.linalg.eig(cov)
    center = np.mean([x, y], axis=1)[:, None]

    # theta = np.degrees(np.arctan2(*rot[:, 0][::-1]))
    # # Width and height are "full" widths, not radius
    # width, height = 2 * z_quant * np.sqrt(val)
    # ellip = Ellipse(xy=center, width=width, height=height, angle=theta)
    # ax.add_artist(ellip)

    t = np.linspace(0, 2.0 * np.pi, 1000)
    xy = np.stack((np.cos(t), np.sin(t)), axis=-1)
    ax.plot(*(rot @ (z_quant * np.sqrt(val) * xy).T + center), c=color, linestyle=linestyle)
    if plot_center:
        ax.scatter(center[0], center[1], c=color, s=size, marker='+')


def plot_orig_image(ax, image, contours=None, points=None):
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap=plt.cm.Spectral)
    if contours is not None:
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], color='w', linewidth=3, markersize=12, zorder=1)
    if points is not None:
        ax.scatter(points[0][1], points[0][0], s=120, marker='X', color='red', zorder=2)
        ax.scatter(points[1][1], points[1][0], s=120, marker='X', color='red', zorder=3)
        ax.annotate("A", (points[0][1], points[0][0]), xytext=(10, 10),
                    textcoords='offset points', color='red', fontsize=50)
        ax.annotate("B", (points[1][1], points[1][0]), xytext=(10, 10),
                    textcoords='offset points', color='red', fontsize=50)


def plot_registered_curve(ax, res, image_shape):
    ax.plot(res[:, 1], res[:, 0], '-r', linewidth=3)
    ax.set_box_aspect(image_shape[0] / image_shape[1])
    ax.set_ylim(ymin=0)
    ax.axis('equal')


def plot_data(ax, ef_coefs, colors, sample_name, random_state):
    # read training data
    EF_data = pd.read_csv('data/pred_ef.csv', header=0)
    dataset = EF_data.iloc[:, np.r_[1:29]]  # elliptic Fourier coefficients
    pca = make_pipeline(PCA(n_components=7, random_state=random_state))
    y = EF_data[["sex"]].iloc[:, 0] 
    pca.fit(dataset, y)
    X_embedded = pca.transform(dataset)
    X_female = X_embedded[y == 'F']
    X_male = X_embedded[y == 'M']
    new_data_embedded = pca.transform(ef_coefs.reshape(1, -1))

    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], s=100,
                    hue=y, hue_order=['F', 'M'], palette=colors)
    # Add confidence ellipses
    draw_conf_ellipse(X_female[:, 0], X_female[:, 1], color=colors['F'], ax=ax)
    draw_conf_ellipse(X_male[:, 0], X_male[:, 1], color=colors['M'], ax=ax)
    ax.set_xlabel(f"PC 1 ({pca.named_steps.pca.explained_variance_ratio_[0] * 100:.2f}%)", fontsize=13)
    ax.set_ylabel(f"PC 2 ({pca.named_steps.pca.explained_variance_ratio_[1] * 100:.2f}%)", fontsize=13)
    sns.scatterplot(x=new_data_embedded[:, 0], y=new_data_embedded[:, 1], s=200,
                    color='black', marker='X', label=sample_name, ax=ax)