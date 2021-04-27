import collections
import contextlib
import importlib.util

import numpy as np
import pandas as pd
import torch


def exporter():
    """Export utility modified from https://stackoverflow.com/a/41895194
    Returns export decorator, __all__ list
    """
    all_ = []

    def decorator(obj):
        all_.append(obj.__name__)
        return obj

    return decorator, all_


export, __all__ = exporter()
__all__.append('exporter')


@export
def sample_1d(f, grid, size=1000):
    """Samples values from a 1d distribution

    Args:
        f: function taking one argument, returning density; not necessarily normalized.
        grid: numpy array of values at which the function is evaluated.
        size: integer, number of samples to pick.

    Returns: numpy array of samples drawn from f
    """
    # Find grid bin size. Can't just do np.diff, it shortens array
    bin_size = np.interp(x=grid,
                         xp=(grid[:-1]+grid[1:])/2,
                         fp=np.diff(grid))
    # Get CDF, then lookup random numbers in inverted CDF
    cdf = np.cumsum(f(grid) * bin_size)
    return np.interp(x=np.random.rand(size),
                     xp=cdf / cdf[-1],
                     fp=grid)


@export
@contextlib.contextmanager
def temp_numpy_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


@export
def load_py_file(path, module_name):
    """Load .py file from path, return as a module named module_name"""
    # From https://stackoverflow.com/questions/67631
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@export
def double_matmul_batch(vec, mat):
    """Does vec @ mat @ vec when both have a batch dimension."""
    n_batch, n = vec.shape
    return torch.matmul(
        vec.reshape(n_batch, 1, n),
        torch.matmul(
            mat,
            vec.reshape(n_batch, n, 1))).squeeze(1).squeeze(1)


@export
def logit(x):
    """Return inverse sigmoid of x"""
    # https://github.com/pytorch/pytorch/issues/37060
    return torch.log(x) - torch.log1p(-x)


@export
def cov_to_std(cov):
    """Return (std errors, correlation coefficent matrix)
    given covariance matrix cov
    """
    std_errs = np.diag(cov) ** 0.5
    corr = cov * np.outer(1 / std_errs, 1 / std_errs)
    return std_errs, corr


@export
def weighted_mean(x, w=None):
    return np.average(x, weights=w)


@export
def weighted_covariance(x, y, w=None):
    return np.average(  (x - weighted_mean(x, w))
                      * (y - weighted_mean(y, w)),
                      weights=w)


@export
def weighted_correlation(x, y, w=None):
    return weighted_covariance(x, y, w) / np.sqrt(
        weighted_covariance(x, x, w) * weighted_covariance(y, y, w))


@export
def linear_fit(x, y, w=None):
    slope, intercept = np.polyfit(x, y, w=w, deg=1)
    fit = np.poly1d([slope, intercept])
    return fit, (slope, intercept)


@export
def to_str_tuple(x):
    """Convert a strings or sequence of strings to a tuple of strings"""
    if isinstance(x, str):
        return (x,)
    elif isinstance(x, list):
        return tuple(x)
    elif isinstance(x, tuple):
        return x
    elif isinstance(x, pd.Series):
        return tuple(x.values.tolist())
    elif isinstance(x, np.ndarray):
        return tuple(x.tolist())
    raise TypeError(f"Expected string or tuple of strings, got {type(x)}")


@export
def flatten_dict(d, separator='_', keep=tuple(), _parent_key=''):
    """Flatten nested dictionaries into a single dictionary,
    indicating levels by separator.
    Stolen from http://stackoverflow.com/questions/6027558
    :param keep: key or list of keys whose values should not be flattened.
    """
    keep = to_str_tuple(keep)
    items = []
    for k, v in d.items():
        new_key = _parent_key + separator + k if _parent_key else k
        if isinstance(v, collections.abc.MutableMapping) and k not in keep:
            items.extend(flatten_dict(v,
                                      separator=separator,
                                      _parent_key=new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


@export
def soft_clip_max(x: torch.Tensor, clip_start):
    """Return x with values above clip_start sigmoid-suppressed.
    """
    # sigmoid(0) = 1/2, sigmoid'(0) = 1/4, so to be smooth at 0...
    return torch.where(
        x > clip_start,
        clip_start + (4 * torch.sigmoid(x - clip_start) - 2),
        x)


@export
def cov_to_std(cov):
    """Return (std errors, correlation coefficent matrix)
    given covariance matrix cov
    """
    std_errs = np.diag(cov) ** 0.5
    corr = cov * np.outer(1 / std_errs, 1 / std_errs)
    return std_errs, corr