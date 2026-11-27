import numpy as np
import pandas as pd
import string
import warnings
from pathlib import Path
import scipy
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.decomposition import PCA

from utils.ef_functions import iefourier

def draw_conf_ellipse(x, y, ax=None, conf=0.95, plot_center=True, ellipse_kwargs=None, center_kwargs=None):
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
    ellipse_kwargs = ellipse_kwargs or {}
    center_kwargs = center_kwargs or {}
    
    ellipse_scale = np.sqrt(scipy.stats.chi2.ppf(conf, df=2))
    cov = np.cov(x, y)
    val, rot = np.linalg.eigh(cov)
    radii = ellipse_scale * np.sqrt(np.maximum(val, 0.0))
    center = np.array([np.mean(x), np.mean(y)])

    t = np.linspace(0, 2.0 * np.pi, 1000)
    xy = np.stack((np.cos(t), np.sin(t)), axis=-1)
    pts = (rot @ (radii * xy).T).T + center
    ax.plot(pts[:, 0], pts[:, 1], **ellipse_kwargs)
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
        pts = np.asarray(points)
        ax.scatter(pts[:, 1], pts[:, 0], s=150, marker="X", color="red", zorder=2)
        for label, (y, x) in zip(string.ascii_uppercase, pts):
            ax.annotate(label, (x, y), xytext=(10, 10), textcoords="offset points", color="red", fontsize=24)

    return ax


def plot_registered_curve(res, ax=None):
    """
    Plot the registered curve.

    Parameters
    ----------
    res : numpy.ndarray
        The registered curve coordinates, array of (row, col) / (y, x) arrays.
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
    ax.set_aspect("equal", adjustable="box")
    ax.set_ylim(ymin=0)

    return ax


## reverse transform min and max PCs
def shape_from_PC(PC_value: float, PC_index: int, pca_class) -> np.ndarray:
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
    shape_PC : np.ndarray
        The reconstructed shape as a np.ndarray of (row, col) / (y, x) coordinates.
    """
    X_PC = np.zeros((1, pca_class.n_components_))
    X_PC[0, PC_index] = PC_value
    X_reconstructed = pca_class.inverse_transform(X_PC).ravel()

    an, bn, cn, dn = np.split(X_reconstructed, 4)
    return iefourier(an, bn, cn, dn)

    
def plot_data(
        fig, 
        sample_ef_coefs: np.ndarray, 
        colors: dict[str, str], 
        sample_name: str, 
        random_state: int = 42,
        data_path: str | Path = "data/pred_ef.csv",
        pca: PCA | None = None,
        ef_data: pd.DataFrame | None = None
):
    """
    Plot the first two PCs of the Elliptic Fourier coefficients from the training dataset pred_ef.csv
    with confidence ellipses separate for males and females. Add current datapoint to the plot as a black X marker.
    Also plot the shapes corresponding to the 0.01 and 0.99 quantiles of the first two PCs alongside x and y axes.

    Parameters
    ---------- 
    fig : matplotlib.figure.Figure
        The figure object to plot on.
    sample_ef_coefs : numpy.ndarray
        The Elliptic Fourier coefficients of the current sample, size (4*k,) = k harmonics.
    colors : dict
        A palette mapping 'F' and 'M' to their respective colors for plotting.
    sample_name : str
        The name of the current sample to add to a legend.
    random_state : int
        The random state for reproducibility.
    data_path: str = "data/pred_ef.csv"
        Location of trainig data.
    pca: PCA | None = None

    ef_data: pd.DataFrame | None = None


    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object with the plotted data.
    """
    ## read training data
    if ef_data is None:
        ef_data = pd.read_csv(data_path)

    sample_ef_coefs = np.asarray(sample_ef_coefs, dtype=float).ravel()
    n_features = len(sample_ef_coefs)

    # Find columns with int numbers as name, these are EF coefficients.
    coef_cols = sorted(
    (col for col in ef_data.columns if str(col).isascii() and str(col).isdigit()),
    key=int
    )

    if len(coef_cols) == 0 or len(coef_cols) % 4 != 0:
        raise ValueError(
            f"Expected EF coefficient columns in `ef_data` to be a non-zero multiple of 4, found {len(coef_cols)}."
        )
    if n_features == 0 or n_features % 4 != 0:
        raise ValueError(
            f"Expected `sample_ef_coefs` length to be a non-zero multiple of 4, got {n_features}."
        )

    n_harmonics_data = len(coef_cols) // 4
    n_harmonics_sample = n_features // 4

    if n_harmonics_sample > n_harmonics_data:
        warnings.warn(
            f"`sample_ef_coefs` has {n_harmonics_sample} harmonics, which exceeds the "
            f"{n_harmonics_data} harmonics available in `ef_data`; truncating `sample_ef_coefs` to {n_harmonics_data}.",
            UserWarning,
            stacklevel=2,
        )
        n_harmonics = n_harmonics_data
        sample_ef_coefs = np.concatenate(
            [block[:n_harmonics] for block in np.split(sample_ef_coefs, 4)]
        )
    elif n_harmonics_sample < n_harmonics_data:
        warnings.warn(
            f"`ef_data` has {n_harmonics_data} harmonics, which exceeds the "
            f"{n_harmonics_sample} harmonics in `sample_ef_coefs`; truncating `ef_data` to {n_harmonics_sample}.",
            UserWarning,
            stacklevel=2,
        )
        n_harmonics = n_harmonics_sample
    else:
        n_harmonics = n_harmonics_data

    selected_cols = [
        col
        for block in range(4)
        for col in coef_cols[block * n_harmonics_data : block * n_harmonics_data + n_harmonics]
    ]
    
    dataset = ef_data[selected_cols]
    y = ef_data["sex"]

    if pca is None:
        pca = PCA(n_components=2, random_state=random_state)
        X_embedded = pca.fit_transform(dataset)
    else:
        X_embedded = pca.transform(dataset)
    
    # PCA transform of the current sample
    new_data_embedded = pca.transform(sample_ef_coefs.reshape(1, -1))

    gs = gridspec.GridSpec(5, 5, figure=fig)
    ax1 = fig.add_subplot(gs[0:4, 1:5])  # Main PC scatter plot

    # Marginal shape thumbnails: (GridSpec slot, PC index, quantile)
    marginal_specs = [
        (gs[0, 0], 1, 0.99),  # PC2 max
        (gs[3, 0], 1, 0.01),  # PC2 min
        (gs[4, 4], 0, 0.99),  # PC1 max
        (gs[4, 1], 0, 0.01),  # PC1 min
    ]

    for spec, pc_idx, q in marginal_specs:
        ax_sub = fig.add_subplot(spec)
        pc_val = np.quantile(X_embedded[:, pc_idx], q)
        shape = shape_from_PC(pc_val, pc_idx, pca)

        ax_sub.plot(shape[:, 1], shape[:, 0], color="black", linewidth=1.5)
        ax_sub.set_aspect("equal")
        ax_sub.axis("off")

    # Plot datapoints from the training sample
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], s=200,
                    hue=y, hue_order=['F', 'M'], palette=colors, alpha=0.7, ax=ax1)

    # Plot the current sample as a black X marker
    sns.scatterplot(x=new_data_embedded[:, 0], y=new_data_embedded[:, 1], s=300,
                    color='black', marker='X', label=sample_name, ax=ax1)

    # Add sex-specific confidence ellipses
    for group in ("F", "M"):
        group_pts = X_embedded[y == group]
        draw_conf_ellipse(
            group_pts[:, 0],
            group_pts[:, 1],
            ax=ax1,
            ellipse_kwargs={"color": colors[group], "linewidth": 2},
            center_kwargs={"color": colors[group], "s": 200, "marker": "+", "linewidth": 3},
        )

    var_ratio = pca.explained_variance_ratio_ * 100
    ax1.set_xlabel(f"PC 1 ({var_ratio[0]:.2f}%)", fontsize=22)
    ax1.set_ylabel(f"PC 2 ({var_ratio[1]:.2f}%)", fontsize=22)
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.legend(fontsize=20)
    ax1.tick_params(axis="both", labelsize=15)

    return fig

