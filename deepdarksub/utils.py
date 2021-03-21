import contextlib
import importlib.util

import numpy as np


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