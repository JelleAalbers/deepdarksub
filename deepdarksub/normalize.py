import numpy as np
from scipy import stats
import torch

from .utils import exporter
export, __all__ = exporter()


@export
class Normalizer:

    def __init__(self, meta, fit_parameters):
        """Initialize scales and offsets from metadata for fit_parameres
        :param meta: DataFrame with manada metadata
        :param fit_parameters: iterable over strings, parameters to fit
        """

        # Estimate distributions of parameters
        # Using a Gaussian for now;
        # would using the actual config distribution be better?
        self.fit_parameters = fit_parameters
        self.dists = {
            p: stats.norm(np.mean(meta[p]), np.std(meta[p]))
            for p in fit_parameters}

    def norm(self, x, param_name, _reverse=False):
        """Normalize x values representing param_name"""
        dist = self.dists[param_name]
        if _reverse:
            return x * dist.std() + dist.mean()
        else:
            return (x - dist.mean()) / dist.std()

    def unnorm(self, x, param_name):
        """Restore physical scale for x values representing param_name"""
        return self.norm(x, param_name, _reverse=True)

    def decode(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        if len(x.shape) == 1:
            # Single item
            x = x.reshape(1, -1)
        n_params = len(self.fit_parameters)
        has_uncertainty = x.shape[1] == 2 * n_params
        y_pred = {}
        y_unc = {}
        for i, p in enumerate(self.fit_parameters):
            y_pred[p] = self.unnorm(x[:, i], p)
            if has_uncertainty:
                y_unc[p] = 2 ** x[:, n_params + i] * self.dists[p].std()
        return y_pred, y_unc
