import numpy as np
import torch

import deepdarksub as dds

export, __all__ = dds.exporter()


@export
class Normalizer:

    def __init__(self, meta=None, fit_parameters=None, means=None, scales=None):
        """Initialize scales and offsets for fit_parameres
        :param meta: DataFrame with manada metadata for a dataset (from csv)
        :param fit_parameters: iterable over strings, parameters to fit
        :param means: means to use, if meta is None
        :param scales: scales to use, if meta is None
        """
        if fit_parameters is None:
            # Should have been a positional argument... backward compatibility
            raise ValueError("Must specify fit_parameters")
        self.fit_parameters = fit_parameters

        if meta is None:
            assert means is not None, "Provide either meta or means/scales"
            # Only keep means/scales for parameters we are fitting
            self.means = {p: means[p] for p in fit_parameters}
            self.scales = {p: scales[p] for p in fit_parameters}

        else:
            self.means = {p: np.mean(meta[p].values) for p in fit_parameters}
            self.scales = {p: np.std(meta[p].values) for p in fit_parameters}

            # For two-component parameters that rotate during augmentation,
            # apply a single scale and no mean shift.
            # This way the rotation transform remains simple, and often
            # these have a mean near 0 and equal x/y distributions anyway.
            for (p1, p2) in (('e1', 'e2'),
                             ('gamma1', 'gamma2'),
                             ('center_x', 'center_y')):
                self.means[p1] = self.means[p2] = 0
                self.scales[p1] = self.scales[p2] = \
                    (self.scales.get(p1, 1) + self.scales.get(p2, 1)) / 2

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
        :param uncertainty: depends on which uncertainty scheme was used;
         - None: will be None
         - diagonal: (dict with) standard deviation for each parameter
         - correlated: covariance matrix (never as dict)

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
            x_pred = self.array_to_dict(x_pred)

        if not uncertainty:
            return x_pred, None
        elif uncertainty == 'correlated':
            # Recover covariance matrix
            _, L, _ = dds.x_to_xp_L(x, n_params)
            scale_vec = np.array([self.scales[p] for p in self.fit_parameters])
            cov = dds.L_to_cov(L) * np.outer(scale_vec, scale_vec)[None, :, :]
            return x_pred, cov
        elif uncertainty == 'diagonal':
            # Recover uncertainty
            x_unc = np.stack([2**x[:, n_params + i] * self.scales[p]
                              for i, p in enumerate(self.fit_parameters)],
                             axis=1)
            if as_dict:
                x_unc = self.array_to_dict(x_unc)
            return x_pred, x_unc
        # else can't happen, n_out would have already crashed

    def array_to_dict(self, x):
        return {p: x[:, i]
                for i, p in enumerate(self.fit_parameters)}
