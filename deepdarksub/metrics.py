import numpy as np
from scipy import stats

from fastai.metrics import AccumMetric

import deepdarksub as dds

export, __all__ = dds.exporter()


class MyMetric:
    suffix = str

    def __init__(self, normalizer, parameter, name, uncertainty):
        self.parameter = parameter
        self.normalizer = normalizer
        self.uncertainty = uncertainty
        self.name = name + '_' + self.suffix
        self.n_params = len(self.normalizer.fit_parameters)

    def make(self):
        def f(x, y):
            try:
                return self._compute(x, y)
            # Metrics shouldn't be able to stop training
            except Exception as e:
                print(f"Caught exception in metric!: {e}")
                return float('nan')
        f.__name__ = self.name
        return AccumMetric(f, to_np=True, flatten=False)

    def _compute(self, y_pred, y_true):
        # Unnormalize, and split off the uncertainty.
        #  * Fastai uses y_pred, y_true convention

        if self.uncertainty == 'diagonal':
            y_pred, y_unc = self.normalizer.decode(
                y_pred[:, :2 * self.n_params],
                uncertainty='diagonal',
                as_dict=True)
        else:
            # Put NaNs for the uncertainty
            y_pred, _ = self.normalizer.decode(
                y_pred[:, :self.n_params],
                as_dict=True)
            y_unc = {k: np.nan * v
                     for k, v in y_pred.items()}

        y_true, _ = self.normalizer.decode(
            y_true[:, :self.n_params],
            as_dict=True)
        y_pred, y_true, y_unc = [
            self.extract_param(y)
            for y in (y_pred, y_true, y_unc)]
        return self.compute(y_true=y_true, y_pred=y_pred, y_unc=y_unc)

    def extract_param(self, y):
        return y[self.parameter]

    def compute(self, *, y_true, y_pred, y_unc):
        raise NotImplementedError


@export
class RMSEMetric(MyMetric):
    suffix = 'rmse'
    def compute(self, *, y_true, y_pred, y_unc):
        return np.std(y_true) / np.std(y_true - y_pred)


@export
class UncertaintyMetric(MyMetric):
    suffix = 'unc'
    def compute(self, *, y_true, y_pred, y_unc):
        return np.std(y_true) / np.mean(y_unc)


@export
class PearsonRMetric(MyMetric):
    suffix = 'rho'
    def compute(self, *, y_true, y_pred, y_unc):
        return stats.pearsonr(y_true, y_pred)[0]


@export
class CorrSSubMetric(MyMetric):
    """Measures correlation of predicted sigma_sub with true [X]"""
    suffix = 'sigma_sub_rho'

    # Don't limit to one parameter, we need a cross-correlation
    def extract_param(self, y):
        return y

    def compute(self, *, y_true, y_pred, y_unc):
        return stats.pearsonr(y_pred['sigma_sub'],
                              y_true[self.parameter])[0]


@export
def all_metrics(fit_parameters, normalizer, short_names, uncertainty):
    if uncertainty == 'diagonal':
        metric_classes = [RMSEMetric, UncertaintyMetric, PearsonRMetric]
    else:
        metric_classes = [RMSEMetric, PearsonRMetric]

    if 'sigma_sub' in fit_parameters:
        metric_classes.append(CorrSSubMetric)

    return [
        f(normalizer, param, short_name, uncertainty).make()
        for f in metric_classes
        for param, short_name in zip(fit_parameters, short_names)]
