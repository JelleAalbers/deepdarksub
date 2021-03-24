import manada
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

import deepdarksub as dds
export, __all__ = dds.exporter()


@export
def image_grid(
        shape, pixel_width=manada.Sources.cosmos.HUBBLE_ACS_PIXEL_WIDTH,
        x0=0, y0=0, edges=True):
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
               vmin=None, vmax=None, **kwargs):

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

    # Plot image, note transposition
    plt.pcolormesh(
        *image_grid(img.shape, pixel_width),
        img.clip(vmin, None).T,
        **kwargs)
    if adjust_ax:
        plt.gca().set_aspect('equal')

    if colorbar:
        # Plot colorbar
        cbar = plt.colorbar(label=label, extend='both')
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
def r_label(x, y, result_name='',
            c='k', fontsize=8, fit_line_style=None,
            side='right'):
    if fit_line_style is None:
        fit_line_style = dict()
    fit_line_style = {'c': c, 'alpha': 0.2, 'linestyle': '--',
                      **fit_line_style}

    fit, (slope, intercept) = dds.linear_fit(x, y)

    # Plot linear fit, preserving lims
    ylim = plt.ylim()
    _x = np.linspace(*plt.xlim(), num=100)
    plt.plot(_x, fit(_x), **fit_line_style)
    plt.ylim(*ylim)

    r = stats.pearsonr(x, y)[0]
    if side == 'left':
        position = dict(x=0.02, y=0.02, ha='left')
    else:
        position = dict(x=0.98, y=0.02, ha='right')
    plt.text(s=(f"Slope = {slope:0.3f}, $R^2 = {r ** 2 * 100 :0.1f}$%"
                + ("\n" + result_name if result_name else '')),
             c=c, **position,
             va='bottom', transform=plt.gca().transAxes,
             fontsize=fontsize)

    return r, fit
