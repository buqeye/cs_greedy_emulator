import numpy as np

latex_markers = ["$\medblackstar$",
"$\medblackdiamond$",
"$\medblacksquare$",
"$\medblackcircle$",
"$\medblacktriangledown$",
"$\medblacktriangleleft$",
"$\medblacktriangleright$",
"$\medblacktriangleup$"]

def plot_empirical_saturation(ax=None, facecolor='lightgray', edgecolor='gray', 
                              alpha=0.4, zorder=9, x=(0.164,0.007), y=(-15.86,0.57), **kwargs):
    if ax is None:
        ax = plt.gca()
    from matplotlib.patches import Rectangle
    # From Drischler et al. (2017), arXiv:1710.08220
    n0 = x[0] # fm**-3
    n0_std = x[1] # fm**-3
    y0 = y[0] # MeV
    y0_std = y[1]  # MeV; errors are added linearly
    # y0_std = np.sqrt(0.37 ** 2 + 0.2 ** 2) # MeV; use this to add them in quadrature
    left = n0 - n0_std
    right = n0 + n0_std
    rect = Rectangle(
        (left, y0 - y0_std), width=right - left, height=2 * y0_std,
        facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, zorder=zorder, **kwargs
    )
    ax.add_patch(rect)
    return ax


def confidence_ellipse(mean, cov, ax, n_std=2.0, facecolor='0.6', edgecolor='0.', alpha=1., **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

#     cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        **kwargs)

#     # Calculating the stdandard deviation of x from
#     # the squareroot of the variance and multiplying
#     # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
#     mean_x = np.mean(x)

#     # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
#     mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean[0], mean[1])

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def define_color_latex(cmap_name='tab20'):
    import matplotlib.pyplot as plt
    extended_colors=plt.get_cmap(cmap_name).colors
    for icolor, color in enumerate(extended_colors):
        print("\definecolor{tab20:" + str(icolor) + "}{rgb}{"
              + str(color[0]) + "," + str(color[1]) + ","+ str(color[2]) + "}")
