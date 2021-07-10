from base64 import b32encode
import collections
import contextlib
from hashlib import sha1
import importlib.util
import json
from pathlib import Path
import subprocess

import numba
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



@numba.njit
def set_numba_random_seed(value):
    np.random.seed(value)


@export
@contextlib.contextmanager
def temp_numpy_seed(seed):
    """Temporarily sets the numpy random generator to this seed,
    then return to previous random state.

    The numba random generator is also reseeded after entering and
    leaving the fixed-seed context.
    """

    state = np.random.get_state()
    np.random.seed(seed)

    # I don't know how to retrieve the numba seed...
    # But we can set it:
    set_numba_random_seed(np.random.randint(np.iinfo(np.uint32).max - 1))

    try:
        yield
    finally:
        np.random.set_state(state)

    # Now that we are back in an unknown random state,
    # seed the numba generator again.
    set_numba_random_seed(np.random.randint(np.iinfo(np.uint32).max - 1))


@export
def load_py_file(path, module_name=None):
    """Load .py file from path.

    If module_name given, import as a module named module_name;
    otherwise, exec and return the locals dictionary.
    """
    if module_name is None:
        with open(path) as f:
            code = compile(f.read(), path, 'exec')
        captured_locals = dict()
        exec(code, globals(), captured_locals)
        return captured_locals
    else:
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


@export
class NumpyJSONEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types
    Edited from mpl3d: mpld3/_display.py
    """

    def default(self, obj):
        try:
            iterable = iter(obj)
        except TypeError:
            pass
        else:
            return [self.default(item) for item in iterable]
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@export
def deterministic_hash(thing, length=10):
    """Return a base32 lowercase string of length determined from hashing
    a container hierarchy
    """
    hashable = hashablize(thing)
    jsonned = json.dumps(hashable, cls=NumpyJSONEncoder)
    digest = sha1(jsonned.encode('ascii')).digest()
    return b32encode(digest)[:length].decode('ascii').lower()


@export
def hashablize(obj):
    """Convert a container hierarchy into one that can be hashed.
    See http://stackoverflow.com/questions/985294
    """
    try:
        hash(obj)
    except TypeError:
        if isinstance(obj, dict):
            return tuple((k, hashablize(v)) for (k, v) in sorted(obj.items()))
        elif isinstance(obj, np.ndarray):
            return tuple(obj.tolist())
        elif hasattr(obj, '__iter__'):
            return tuple(hashablize(o) for o in obj)
        else:
            raise TypeError("Can't hashablize object of type %r" % type(obj))
    else:
        return obj


@export
def run_command(command, show_output=True):
    """Run command and show its output in STDOUT"""
    # Is there no easier way??
    with subprocess.Popen(
            command.split(),
            bufsize=1,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT) as p:
        for line in iter(p.stdout.readline, ''):
            if show_output:
                print(line.rstrip())


@export
def make_dummy_dataset(dirname='dummy_dataset', n_images=20):
    """Return Path to dummy dataset with noise images and meaningless
    metadata
    """
    folder = Path(dirname)
    folder.mkdir(exist_ok=True)

    dummy_image = np.random.normal(
        0.2, 0.01, size=(64, 64)).astype(np.float32)
    for i in range(n_images):
        np.save(Path(dirname) / f'image_{i:07d}', dummy_image)

    dummy_meta = {
        'main_deflector_parameters_theta_E': 1.,
        'subhalo_parameters_sigma_sub': 0.1,
        'los_parameters_delta_los': 0.1,
        'main_deflector_parameters_center_x': 0.01,
        'main_deflector_parameters_center_y': 0.01,
        'main_deflector_parameters_gamma': 2.,
        'main_deflector_parameters_gamma1': 0.1,
        'main_deflector_parameters_gamma2': 0.1,
        'main_deflector_parameters_e1': 0.1,
        'main_deflector_parameters_e2': 0.1,
        'source_parameters_z_source': 1.}
    df = pd.DataFrame(
        {k: np.random.normal(1., .1, size=n_images) * v
         for k, v in dummy_meta.items()})
    df['source_parameters_catalog_i'] = np.random.choice(
        # 548 is in val_galaxies, 0 is not
        # This way we will have a train and val dataset.
        [0, 548], n_images)
    df.to_csv(Path(dirname) / 'metadata.csv')
    return folder


@export
def spy_on_method(object, method_name, results_dict, alias):
    """Monkeypatches object.method_name so that each call will append
    (args, kwargs, result) to results_dict[alias]
    """
    if alias is None:
        alias = method_name
    results_dict.setdefault(alias, [])
    orig_f = getattr(object, method_name)
    def spied_f(*args, **kwargs):
        result = orig_f(*args, **kwargs)
        results_dict[alias].append((args, kwargs, result))
        return result
    setattr(object, method_name, spied_f)

