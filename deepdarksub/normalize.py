import numpy as np
import torch

import deepdarksub as dds

export, __all__ = dds.exporter()


@export
class Normalizer:

    def __init__(self, meta, fit_parameters):
        """Initialize scales and offsets from metadata for fit_parameres
        :param meta: DataFrame with manada metadata
        :param fit_parameters: iterable over strings, parameters to fit
        """
        self.fit_parameters = fit_parameters
        self.means = {p: np.mean(meta[p]) for p in fit_parameters}
        self.scales = {p: np.std(meta[p]) for p in fit_parameters}

    def norm(self, x, param_name, _reverse=False):
        """Normalize x values representing param_name"""
        if _reverse:
            return x * self.scales[param_name] + self.means[param_name]
        else:
            return (x - self.means[param_name]) / self.scales[param_name]

    def unnorm(self, x, param_name):
        """Restore physical scale for x values representing param_name"""
        return self.norm(x, param_name, _reverse=True)

    def decode(self, x, uncertainty=False, as_dict=True):
        """Return (prediction, uncertainty) given neural net output x
        :param x: torch Tensor (n_images, n_outputs)
        :param uncertainty: which uncertainty scheme was used;

        """
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        if len(x.shape) == 1:
            # Single item
            x = x.reshape(1, -1)
        n_params = len(self.fit_parameters)
        assert x.shape[1] == dds.n_out(n_params, uncertainty), \
            "Output does not match uncertainty"

        # Recover best-fit parameters
        x_pred = np.stack([self.unnorm(x[:, i], p)
                           for i, p in enumerate(self.fit_parameters)],
                          axis=1)
        if as_dict:
            x_pred = self._array_to_dict(x_pred)

        if not uncertainty:
            return x_pred, None
        elif uncertainty == 'correlated':
            # Recover covariance matrix
            _, L = dds.x_to_xp_L(x, 2)
            scale_vec = np.array([self.scales[p] for p in self.fit_parameters])
            cov = dds.L_to_cov(L) * np.outer(scale_vec, scale_vec)[None, :, :]
            return x_pred, cov
        elif uncertainty == 'diagonal':
            # Recover uncertainty
            x_unc = np.stack([2 ** x[:, n_params + i] * self.scales[p]
                              for i, p in enumerate(self.fit_parameters)],
                             axis=1)
            if as_dict:
                x_unc = self._array_to_dict(x_unc)
            return x_pred, x_unc
        # else can't happen, n_out would have already crashed

    def _array_to_dict(self, x):
        return {p: x[:, i]
                for i, p in enumerate(self.fit_parameters)}
