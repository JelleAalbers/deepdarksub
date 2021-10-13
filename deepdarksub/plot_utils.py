import manada
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

import deepdarksub as dds
export, __all__ = dds.exporter()

__all__.extend(['parameter_colors', 'parameter_labels'])

# Common colors and labels for parameters
parameter_colors = {
    'theta_E': 'g',
    'sigma_sub': 'b',
    'delta_los': 'darkorange',
    'shmf_plaw_index': 'saddlebrown',
    'center_x': 'magenta', 'center_y': 'magenta',
    'gamma': 'purple',
    'e1': 'crimson', 'e2': 'crimson',
    'gamma1': 'cyan', 'gamma2': 'cyan'}
parameter_labels = {
    'theta_E': r'$\theta_E$',
    'sigma_sub': r'$\Sigma_\mathrm{sub}$',
    'shmf_plaw_index': r'$\gamma_\mathrm{SHMF}$',
    'delta_los': r'$\delta_\mathrm{los}$',
    'center_x': r'$x$', 'center_y': r'$y$',
    'gamma': r'$\gamma$',
    'e1': r'$e_1$', 'e2': r'$e_2$',
    'gamma1': r'$\gamma_1$', 'gamma2': r'$\gamma_2$'}


@export
def save_plot(*plot_name, pdf=False, **kwargs):
    """Save plot to ./plots/plot_name.png with standard options
        If several name elements passed, join by _.
    """
    kwargs.setdefault('bbox_inches', 'tight')
    kwargs.setdefault('dpi', 200)
    plot_name = '_'.join(plot_name)
    plt.savefig(f'./plots/{plot_name}.png', **kwargs)
    if pdf:
        plt.savefig(f'./plots/{plot_name}.pdf', **kwargs)


@export
def image_grid(
        shape, pixel_width=manada.Sources.cosmos.HUBBLE_ACS_PIXEL_WIDTH,
        x0=0, y0=0, edges=True):
    """Return x, y edges of image coordinates

    Args:
        - pixel_width: pixel width (in whatever units you want, e.g. arcsec)
        - x0, y0: center of grid
        - edges: If True (default), returns pixel edges.
            If False, returns pixel centers
    """
    nx, ny = shape
    dx = nx * pixel_width
    dy = nx * pixel_width
    extra = 1 if edges else 0
    x = np.linspace(-dx / 2, dx / 2, nx + extra) + x0
    y = np.linspace(-dy / 2, dy / 2, ny + extra) + y0
    return x, y


@export
def plot_image(img,
               pixel_width=manada.Sources.cosmos.HUBBLE_ACS_PIXEL_WIDTH,
               log_scale=True,
               label='HST F814W',
               colorbar=True,
               adjust_ax=True,
               vmin=None, vmax=None,
               cbar_kwargs=None,
               **kwargs):

    # Set reasonable defaults
    if vmax is None:
        vmax = img.max()
    if vmin is None:
        if log_scale:
            vmin = vmax * 1e-3
        else:
            vmin = img.min()
    kwargs.setdefault(
        'norm',
        matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        if log_scale else matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
    kwargs.setdefault('cmap', 'magma')
    if cbar_kwargs is None:
        cbar_kwargs = dict()

    # Plot image, note transposition
    plt.pcolormesh(
        *image_grid(img.shape, pixel_width),
        img.clip(vmin, None).T,
        **kwargs)
    if adjust_ax:
        plt.gca().set_aspect('equal')

    if colorbar:
        # Plot colorbar
        cbar = plt.colorbar(label=label, extend='both', **cbar_kwargs)
        cax = cbar.ax
        # Format colobar ticks as scalars (not 10^x)
        cax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
        if len(cbar.get_ticks()) < 2:
            # If there are very few colorbar ticks, show minor ticks too
            cax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
            cax.tick_params(axis='y', labelsize=7, which='minor')

    plt.xlabel("[Arcseconds]")
    plt.ylabel("[Arcseconds]")


@export
def show_image(*args, **kwargs):
    # Alias for plot_image, since I keep mistyping it
    return plot_image(*args, **kwargs)


@export
def r_label(x, y, w=None, result_name='',
            c='k', fontsize=8, fit_line_style=None,
            background_alpha=0.4,
            side='right'):
    if fit_line_style is None:
        fit_line_style = dict()
    fit_line_style = {'c': c, 'alpha': 0.2, 'linestyle': '--',
                      **fit_line_style}

    fit, (slope, intercept) = dds.linear_fit(x, y, w)

    # Plot linear fit, preserving lims
    ylim = plt.ylim()
    _x = np.linspace(*plt.xlim(), num=100)
    plt.plot(_x, fit(_x), **fit_line_style)
    plt.ylim(*ylim)

    r = dds.weighted_correlation(x, y, w)
    if side == 'left':
        position = dict(x=0.02, y=0.02, ha='left')
    else:
        position = dict(x=0.98, y=0.02, ha='right')
    plt.text(s=(f"Slope = {slope:0.3f}, $R^2 = {r ** 2 * 100 :0.1f}$%"
                + ("\n" + result_name if result_name else '')),
             c=c, **position,
             va='bottom', transform=plt.gca().transAxes,
             fontsize=fontsize).set_bbox(dict(facecolor='w', alpha=background_alpha, linewidth=0))

    return r, fit


@export
def axis_to_data(x, y, ax=None):
    if ax is None:
        ax = plt.gca()
    return ax.transData.inverted().transform(
        ax.transAxes.transform((x, y)))


@export
def logticks(tmin, tmax=None, tick_at=None):
    if tick_at is None:
        tick_at = (1, 2, 5, 10)
    a, b = np.log10([tmin, tmax])
    a = np.floor(a)
    b = np.ceil(b)
    ticks = np.sort(np.unique(np.outer(
        np.array(tick_at),
        10. ** np.arange(a, b)).ravel()))
    ticks = ticks[(tmin <= ticks) & (ticks <= tmax)]
    return ticks


@export
def log_x(a=None, b=None, scalar_ticks=True, tick_at=None):
    plt.xscale('log')
    if a is not None:
        if b is None:
            a, b = a[0], a[-1]
        plt.xlim(a, b)
        ax = plt.gca()
        if scalar_ticks:
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
            ax.set_xticks(logticks(a, b, tick_at))


@export
def log_y(a=None, b=None, scalar_ticks=True, tick_at=None):
    plt.yscale('log')
    if a is not None:
        if b is None:
            a, b = a[0], a[-1]
        ax = plt.gca()
        plt.ylim(a, b)
        if scalar_ticks:
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
            ax.set_yticks(logticks(a, b, tick_at))


@export
def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.

    Stolen from https://stackoverflow.com/questions/18311909
    """
    if exponent is None:
        exponent = int(np.floor(np.log10(np.abs(num))))
    coeff = round(num / float(10 ** exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits

    return r"${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent, precision)


@export
def confidence_ellipse(
        mean, cov, ax=None, percentile=50,
        **kwargs):
    """Plot and return a covariance confidence ellipse

    Args:
     - mean: array-like, shape (2,); location of mean
     - cov: array-like, shape (2, 2), 2d covariance matrix
     - ax: matplotlib.axes.Axes to draw the ellipse into
     - percentile: Probability mass included in the ellipse;
        50% by default.
     - **kwargs forwarded to `~matplotlib.patches.Ellipse`
    """
    # Adapted from Carsten Schelp / matplotlib example gallery:
    #  - https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
    #  - https://gist.github.com/CarstenSchelp/b992645537660bda692f218b562d0712
    if ax is None:
        ax = plt.gca()
    mean = np.asarray(mean)
    cov = np.asarray(cov)
    assert mean.shape == (2,)
    assert cov.shape == (2,2)
    kwargs.setdefault('edgecolor', 'k')
    kwargs.setdefault('facecolor', 'none')

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = matplotlib.patches.Ellipse(
        (0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
        **kwargs)

    # Scale axes to match probability content. See e.g.
    # https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
    scale = stats.chi2(df=2).ppf(percentile/100)**0.5
    scale_x = np.sqrt(cov[0, 0]) * scale
    scale_y = np.sqrt(cov[1, 1]) * scale

    transf = matplotlib.transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(*mean)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
