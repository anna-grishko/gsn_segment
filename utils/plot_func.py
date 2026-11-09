import numpy as np
import pandas as pd
import scipy
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

from utils.ef_functions import iefourier

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
        ax.scatter(points[0][1], points[0][0], s=150, marker='X', color='red', zorder=2)
        ax.scatter(points[1][1], points[1][0], s=150, marker='X', color='red', zorder=3)
        ax.annotate("A", (points[0][1], points[0][0]), xytext=(10, 10),
                    textcoords='offset points', color='red', fontsize=50)
        ax.annotate("B", (points[1][1], points[1][0]), xytext=(10, 10),
                    textcoords='offset points', color='red', fontsize=50)


def plot_registered_curve(ax, res, image_shape):
    ax.plot(res[:, 1], res[:, 0], '-r', linewidth=3)
    ax.set_box_aspect(image_shape[0] / image_shape[1])
    ax.set_ylim(ymin=0)
    ax.axis('equal')


## reverse transform min and max PCs
def shape_from_PC(PC_value, PC_index, pca_class):
    W = pca_class.components_  # rotation matrix
    n_components = pca_class.n_components_
    X_PC = np.zeros(n_components)
    X_PC[PC_index] = PC_value
    X_reconstructed_PC = np.dot(X_PC, W) + pca_class.mean_
    shape_PC = iefourier(X_reconstructed_PC[0:n_components],
                         X_reconstructed_PC[n_components:2*n_components],
                         X_reconstructed_PC[2*n_components:3*n_components],
                         X_reconstructed_PC[3*n_components:4*n_components], n_harmonics=n_components)
    return shape_PC
    
def plot_data(fig, ef_coefs, colors, sample_name, random_state):
    ## read training data
    EF_data = pd.read_csv('data/pred_ef.csv', header=0)
    dataset = EF_data.iloc[:, np.r_[1:29]]  # elliptic Fourier coefficients
    pca = make_pipeline(PCA(n_components=7, random_state=random_state))
    y = EF_data[["sex"]].iloc[:, 0] 
    pca.fit(dataset, y)
    X_embedded = pca.transform(dataset)
    X_female = X_embedded[y == 'F']
    X_male = X_embedded[y == 'M']
    new_data_embedded = pca.transform(ef_coefs.reshape(1, -1))
    ## reverse transform min and max PCs
    # PC1
    shape_PC1_max = shape_from_PC(np.quantile(X_embedded[:, 0], 0.99), 0, pca.named_steps['pca'])
    shape_PC1_min = shape_from_PC(np.quantile(X_embedded[:, 0], 0.01), 0, pca.named_steps['pca'])
    # PC2
    shape_PC2_max = shape_from_PC(np.quantile(X_embedded[:, 1], 0.99), 1, pca.named_steps['pca'])
    shape_PC2_min = shape_from_PC(np.quantile(X_embedded[:, 1], 0.01), 1, pca.named_steps['pca'])

    gs = gridspec.GridSpec(5, 5, figure=fig)
    ax1 = fig.add_subplot(gs[0:4, 1:5])  # main PCs plot
    ax2 = fig.add_subplot(gs[0, 0])  # PC2 max
    ax3 = fig.add_subplot(gs[3, 0])  # PC2 min
    ax4 = fig.add_subplot(gs[4, 4])  # PC1 max
    ax5 = fig.add_subplot(gs[4, 1])  # PC1 min

    ax4.plot(shape_PC1_max['x'], shape_PC1_max['y'], color='black', linewidth=1)
    ax4.axis("off")
    ax5.plot(shape_PC1_min['x'], shape_PC1_min['y'], color='black', linewidth=1)
    ax5.axis("off")
    ax2.plot(shape_PC2_max['x'], shape_PC2_max['y'], color='black', linewidth=1)
    ax2.axis("off")
    ax3.plot(shape_PC2_min['x'], shape_PC2_min['y'], color='black', linewidth=1)
    ax3.axis("off")

    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], s=200,
                    hue=y, hue_order=['F', 'M'], palette=colors, ax=ax1)
    # Add confidence ellipses
    draw_conf_ellipse(X_female[:, 0], X_female[:, 1], color=colors['F'], ax=ax1)
    draw_conf_ellipse(X_male[:, 0], X_male[:, 1], color=colors['M'], ax=ax1)
    ax1.set_xlabel(f"PC 1 ({pca.named_steps.pca.explained_variance_ratio_[0] * 100:.2f}%)", fontsize=22)
    ax1.set_ylabel(f"PC 2 ({pca.named_steps.pca.explained_variance_ratio_[1] * 100:.2f}%)", fontsize=22)
    sns.scatterplot(x=new_data_embedded[:, 0], y=new_data_embedded[:, 1], s=300,
                    color='black', marker='X', label=sample_name, ax=ax1)
    ax1.legend(fontsize=20)
    ax1.tick_params(axis="x", labelsize=15)
    ax1.tick_params(axis="y", labelsize=15)
