import numpy as np
import torch

import deepdarksub as dds

export, __all__ = dds.exporter()
__all__.extend(['parameter_domains'])

parameter_domains = {
    # Note theta_E, sigma_sub, etc = 0 are disallowed,
    # since it crashes lognormal priors.
    'theta_E': (np.exp(-5), 2.),

    'log_theta_E': (-5, np.log(2)),

    # TODO: 1e-2 here is for the high-mass pivot point... disable for low mass
    'sigma_sub': (-0.006, 0.008),

    'shmf_plaw_index': (-3, -1),

    'log_sigma_sub': (-5, 0),

    'delta_los': (np.exp(-5), 5.),
    'log_delta_los': (-5, np.log(5)),

    'gamma': (1., 3.),
    'log_gamma': (np.log(1), np.log(3)),

    'center_x': (-1, 1),
    'center_y': (-1, 1),
    'gamma1': (-1, 1),
    'gamma2': (-1, 1),
    'e1': (-1, 1),
    'e2': (-1, 1),
}


@export
class Inference:

    def __init__(
            self,
            manada_config=None,
            fit_parameters=None):
        manada_config = dds.load_manada_config(manada_config)
        config_dict = manada_config.config_dict
        self.fit_parameters = fit_parameters

        # Scan manada config for distributions
        self.prior_dists = dict()
        for _, sec in config_dict.items():
            for pname, val in sec['parameters'].items():
                if pname in dds.short_names.values() and callable(val):
                    dist = val.__self__
                    dist_name = dist.dist.name
                    if dist_name in ('norm', 'truncnorm'):
                        # Manada's truncnorms are just norms, the truncation
                        # happens at -7 sigma or something.
                        self.prior_dists[pname] = torch.distributions.Normal(
                            dist.mean(), dist.std())
                    elif dist_name == 'lognorm':
                        # See https://pytorch.org/docs/stable/distributions.html
                        # and especially end of 'notes' on
                        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html
                        mu, sigma = np.log(dist.kwds['scale']), dist.kwds['s']
                        self.prior_dists[pname] = \
                            torch.distributions.LogNormal(mu, sigma)
                        self.prior_dists['log_' + pname] = \
                            torch.distributions.Normal(mu, sigma)
                    else:
                        raise ValueError(f"Unknown distribution: {dist_name}")

    def dict_to_stack(self, x, params=None):
        if params is None:
            params = self.fit_parameters
        return np.stack([x[p] for p in params], axis=1)

    def stack_to_dict(self, x, params=None):
        if params is None:
            params = self.fit_parameters
        return {pname: x[:,i] for i, pname in enumerate(params)}

    def select_params(self, x, params=None):
        if params is None:
            params = self.fit_parameters
        if isinstance(params, str):
            params = [params]
        param_indices = [self.fit_parameters.index(p) for p in params]

        if (params == self.fit_parameters) or (len(params) == x.shape[-1]):
            # Already selected parameters
            return x

        # Downselect parameters
        assert x.shape[-1] == len(self.fit_parameters)
        n_dims = len(x.shape)
        if n_dims == 1 + 1:
            return x[:,param_indices]
        elif n_dims == 1 + 2:
            assert x.shape[-2] == len(self.fit_parameters)
            return select_from_matrix_stack(x, param_indices)
        else:
            raise ValueError("Expected stack of vectors or matrices, "
                                "got shape {x.shape}")

    def prepare_inputs(self, *args, params):
        if params is None:
            params = self.fit_parameters
        return [
            torch.tensor(self.select_params(x, params),
                         dtype=torch.float64)
            for x in args]

    def log_prior(self, truths, params=None):
        return self._log_prior(
            *self.prepare_inputs(truths, params=params),
            params).numpy()

    def _log_prior(self, truths, params=None):
        if params is None:
            params = self.fit_parameters
        result = 0. * truths[:, 0]
        for i, pname in enumerate(params):
            x = truths[:,i]
            result += self.prior_dists[pname].log_prob(x)

        return result

    def log_posterior(self, truths, preds, precs, params=None):
        return self._log_posterior(
            *self.prepare_inputs(truths, preds, precs, params=params)).numpy()

    def _log_posterior(self, truths, preds, precs):
        n_images, n_params = truths.shape
        assert preds.shape == (n_images, n_params)
        assert precs.shape == (n_images, n_params, n_params)

        diff = preds - truths
        return -0.5 * (diff[:,None,:] @ precs @ diff[:,:,None])[:,0,0]

    def log_likelihood(self, truths, preds, precs, params=None):
        """Return array of log likelihood ratios for several images

        Truth-independent terms (e.g. log det sigma) are omitted

        Args:
         - truths, preds: Truths and predictions, (n_images, n_params)
         - precs: Precision matrices, (n_images, n_params, n_params)
         - params: list of parameters to use. If
        """
        return (
            self.log_posterior(truths, preds, precs, params)
            - self.log_prior(truths, params))

    def _log_likelihood(self, truths, preds, precs, params=None):
        return (
            self._log_posterior(truths, preds, precs)
            - self._log_prior(truths, params))

    def find_mles(self, preds, precs,
                  params=None, verbose=True,
                  lr_start=1e-3, lr_stop=1e-7, patience=50,
                  max_iter=30_000, report_each=1000):
        """Return array of maximum likelihood estimates, same shape as preds"""
        if params is None:
            params = self.fit_parameters
        preds_t, precs_t = self.prepare_inputs(preds, precs, params=params)

        # Guess that MLEs are at the predictions (~posterior means)
        mle_t = 0. + preds_t
        mle_t.requires_grad = True
        optimizer = torch.optim.Adam([mle_t], lr=lr_start)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=0.5, patience=patience,
            eps=0.1 * lr_stop, verbose=verbose)
        last_mean_ll = None

        for i in range(max_iter):
            optimizer.zero_grad()

            ll = self._log_likelihood(mle_t, preds_t, precs_t, params=params)

            if i % report_each == 0:
                mean_ll = ll.mean().detach().numpy()
                print(i, mean_ll) if verbose else 0
                if last_mean_ll and last_mean_ll > mean_ll:
                    print(':-(') if verbose else 0
                last_mean_ll = mean_ll
            i += 1

            # Each 'parameter' (MLE estimate) affects just one image.
            # Thus, using sum here ensures each gradient component,
            # and thus the change in its direction, is independent of n_images.
            loss = (-ll).sum()
            loss.backward()
            optimizer.step()

            # Restore to valid domain
            with torch.no_grad():
                for i, pname in enumerate(params):
                    mle_t[:,i].clamp_(*dds.parameter_domains[pname])

            scheduler.step(loss)
            if optimizer.param_groups[0]['lr'] < lr_stop:
                break

        return mle_t.detach().numpy()


def select_from_matrix_stack(matrix, select_i):
    """Select specific simultaneous row and column indices
    from a stack of matrices"""
    sel_x, sel_y = np.meshgrid(select_i, select_i, indexing='ij')
    return (
        matrix[:, sel_x.ravel(), sel_y.ravel()]
        .reshape([-1] + list(sel_x.shape)))

np.testing.assert_equal(
    select_from_matrix_stack(np.arange(32).reshape(2,4,4), [3, 0])[0].ravel(),
    np.array([15, 12, 3, 0]))
