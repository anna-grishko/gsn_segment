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

def draw_conf_ellipse(x, y, ax=None, conf=0.95, plot_center=True, ellipse_kwargs={}, center_kwargs={}):
    """
    Draw normal confidence ellipse for given x and y data points.
    
    Parameters
    ----------
    x : array-like
        X coordinates of the data points.
    y : array-like
        Y coordinates of the data points.
    ax : matplotlib.axes.Axes
        Axis object on which to draw the ellipse.
    conf : float
        Value from 0 to 1 indicating the bivariate normal coverage level.
    plot_center : bool
        Whether to plot the center point of the ellipse.
    ellipse_kwargs : dict
        Keyword arguments for the ellipse line style (e.g., 'color', 'linewidth', 'linestyle').
    center_kwargs : dict
        Keyword arguments for the center point style (e.g., 'color', 's', 'marker').

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis object with the drawn ellipse and center point (if specified).
    """

    if ax is None:
        ax = plt.gca()  # get current axes if None provided
    
    ellipse_scale = np.sqrt(scipy.stats.chi2.ppf(conf, df=2))
    cov = np.cov(x, y)
    val, rot = np.linalg.eig(cov)
    center = np.mean([x, y], axis=1)[:, None]

    t = np.linspace(0, 2.0 * np.pi, 1000)
    xy = np.stack((np.cos(t), np.sin(t)), axis=-1)
    ax.plot(*(rot @ (ellipse_scale * np.sqrt(val) * xy).T + center), **ellipse_kwargs)
    if plot_center:
        ax.scatter(center[0], center[1], **center_kwargs)

    return ax


def plot_orig_image(image, ax=None, contours=None, points=None):
    """
    Plot the original image with optional contours and points.

    Parameters
    ----------
    image : 
        The image to plot.
    ax : matplotlib.axes.Axes
        Axis object on which to draw the ellipse.
    contours : list of numpy.ndarray, optional
        List of contours to plot.
    points : list of numpy.ndarray, optional
        List of points to plot.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis object with the plotted image, contours, and points.
    """

    if ax is None:
        ax = plt.gca()  # get current axes if None provided

    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap=plt.cm.Spectral)
    if contours is not None:
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], color='w', linewidth=3, markersize=12, zorder=1)
    if points is not None:
        ax.scatter(points[0][1], points[0][0], s=150, marker='X', color='red', zorder=2)
        ax.scatter(points[1][1], points[1][0], s=150, marker='X', color='red', zorder=2)
        ax.annotate("A", (points[0][1], points[0][0]), xytext=(10, 10),
                    textcoords='offset points', color='red', fontsize=50)
        ax.annotate("B", (points[1][1], points[1][0]), xytext=(10, 10),
                    textcoords='offset points', color='red', fontsize=50)

    return ax


def plot_registered_curve(res, image_shape, ax=None):
    """
    Plot the registered curve.

    Parameters
    ----------
    res : numpy.ndarray
        The registered curve coordinates, array of [x,y] arrays.
    image_shape : tuple
        The shape of the image.
    ax : matplotlib.axes.Axes, optional
        Axis object on which to draw the curve.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis object with the plotted curve.
    """

    if ax is None:
            ax = plt.gca()  # get current axes if None provided
    
    ax.plot(res[:, 1], res[:, 0], '-r', linewidth=3)
    # set aspect ratio  so image is not distorted
    ax.set_box_aspect(image_shape[0] / image_shape[1]) 
    ax.set_ylim(ymin=0)
    ax.axis('equal')

    return ax


## reverse transform min and max PCs
def shape_from_PC(PC_value, PC_index, pca_class):
    """
    Reconstruct the shape from Elliptic Fourier coefficients
    for a given principal component value from the pca class.

    Parameters
    ----------
    PC_value : float
        The value of the principal component to reconstruct.
    PC_index : int
        The index of the principal component to reconstruct.
    pca_class : sklearn.decomposition.PCA
        The PCA class used for the transformation.

    Returns
    -------
    shape_PC : dict
        The reconstructed shape as a dictionary of x and y coordinates.
    """

    W = pca_class.components_  # rotation matrix
    n_components = pca_class.n_components_
    X_PC = np.zeros(n_components)
    X_PC[PC_index] = PC_value
    # reconstruct EF coefficients from PCA
    X_reconstructed_PC = np.dot(X_PC, W) + pca_class.mean_
    # reconstruct shape coordinates from EF coefficients
    n_coefs = 7 # TODO standartize so any number of harmonics canm be used
    shape_PC = iefourier(X_reconstructed_PC[0:n_coefs],  # A coefficients
                         X_reconstructed_PC[n_coefs:2*n_coefs],  # B coefficients
                         X_reconstructed_PC[2*n_coefs:3*n_coefs],  # C coefficients
                         X_reconstructed_PC[3*n_coefs:4*n_coefs],  # D coefficients
                         n_harmonics=n_coefs)
    
    return shape_PC
    
def plot_data(fig, ef_coefs, colors, sample_name, random_state):
    """
    Plot the first two PCs of the Elliptic Fourier coefficients from the training dataset pred_ef.csv
    with confidence ellipses separate for males and females. Adds current datapoint to the plot as a black X marker.
    Also plots the shapes corresponding to the 0.01 and 0.99 quantiles of the first two PCs alongside x and y axes.

    Parameters
    ---------- 
    fig : matplotlib.figure.Figure
        The figure object to plot on.
    ef_coefs : numpy.ndarray
        The Elliptic Fourier coefficients of the current sample, size (28,) = 7 harmonics.
    colors : dict
        A palette mapping 'F' and 'M' to their respective colors for plotting.
    sample_name : str
        The name of the current sample to add to a legend.
    random_state : int
        The random state for reproducibility.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object with the plotted data.
    """
    ## read training data
    EF_data = pd.read_csv('data/pred_ef.csv', header=0)
    # TODO standartize so any number of harmonics canm be used
    dataset = EF_data.iloc[:, np.r_[1:29]]  # elliptic Fourier coefficients (7 harmonics)
    pca = make_pipeline(PCA(n_components=2, random_state=random_state))
    y = EF_data[["sex"]].iloc[:, 0] 
    pca.fit(dataset, y)
    X_embedded = pca.transform(dataset)
    # prepare data for ellipces plotting
    X_female = X_embedded[y == 'F']
    X_male = X_embedded[y == 'M']
    # PCA transform of the current sample
    new_data_embedded = pca.transform(ef_coefs.reshape(1, -1))
    ## reverse transform min and max PCs
    # PC1 (index=0)
    shape_PC1_max = shape_from_PC(np.quantile(X_embedded[:, 0], 0.99), 0, pca.named_steps['pca'])
    shape_PC1_min = shape_from_PC(np.quantile(X_embedded[:, 0], 0.01), 0, pca.named_steps['pca'])
    # PC2 (index=1)
    shape_PC2_max = shape_from_PC(np.quantile(X_embedded[:, 1], 0.99), 1, pca.named_steps['pca'])
    shape_PC2_min = shape_from_PC(np.quantile(X_embedded[:, 1], 0.01), 1, pca.named_steps['pca'])

    gs = gridspec.GridSpec(5, 5, figure=fig)
    ax1 = fig.add_subplot(gs[0:4, 1:5])  # main PCs plot
    ax2 = fig.add_subplot(gs[0, 0])  # PC2 max
    ax3 = fig.add_subplot(gs[3, 0])  # PC2 min
    ax4 = fig.add_subplot(gs[4, 4])  # PC1 max
    ax5 = fig.add_subplot(gs[4, 1])  # PC1 min

    ax4.plot(shape_PC1_max['x'], shape_PC1_max['y'], color='black')
    ax4.axis("off")
    ax5.plot(shape_PC1_min['x'], shape_PC1_min['y'], color='black')
    ax5.axis("off")
    ax2.plot(shape_PC2_max['x'], shape_PC2_max['y'], color='black')
    ax2.axis("off")
    ax3.plot(shape_PC2_min['x'], shape_PC2_min['y'], color='black')
    ax3.axis("off")

    # plot datapoints from the training sample
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], s=200,
                    hue=y, hue_order=['F', 'M'], palette=colors, alpha=0.7, ax=ax1)
    # Add confidence ellipses
    draw_conf_ellipse(X_female[:, 0], X_female[:, 1], ax=ax1,
                      ellipse_kwargs={'color': colors['F'], 'linewidth': 2},
                      center_kwargs={'color': colors['F'], 's': 200, 'marker': '+', 'linewidth': 3})
    draw_conf_ellipse(X_male[:, 0], X_male[:, 1], ax=ax1,
                      ellipse_kwargs={'color': colors['M'], 'linewidth': 2},
                      center_kwargs={'color': colors['M'], 's': 200, 'marker': '+', 'linewidth': 3})
    ax1.set_xlabel(f"PC 1 ({pca.named_steps.pca.explained_variance_ratio_[0] * 100:.2f}%)", fontsize=22)
    ax1.set_ylabel(f"PC 2 ({pca.named_steps.pca.explained_variance_ratio_[1] * 100:.2f}%)", fontsize=22)
    # plot the current sample as a black X marker
    sns.scatterplot(x=new_data_embedded[:, 0], y=new_data_embedded[:, 1], s=300,
                    color='black', marker='X', label=sample_name, ax=ax1)
    ax1.legend(fontsize=20)
    ax1.tick_params(axis="x", labelsize=15)
    ax1.tick_params(axis="y", labelsize=15)

    return fig

