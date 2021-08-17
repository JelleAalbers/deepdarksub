import numpy as np

import deepdarksub as dds



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
        for _, sec in config_dict.values():
            for pname, val in sec['parameters'].items():
                if pname in dds.short_names and callable(val):
                    self.prior_dists[pname].append(val.__self__)

    def select_params(self, *args, params=None):
        param_indices = [self.fit_parameters.index(p) for p in params]

        y = []
        for x in args:
            if (params == self.fit_parameters) or (len(params) == x.shape[-1]):
                # Already selected parameters
                y.append(x)
                continue

            # Downselect parameters
            assert x.shape[-1] == len(self.fit_parameters)
            n_dims = len(x.shape)
            if n_dims == 1 + 1:
                y.append( x[:,param_indices] )
            elif n_dims == 1 + 2:
                assert x.shape[-2] == len(self.fit_parameters)
                y.append( select_from_matrix_stack(x, param_indices) )
            else:
                raise ValueError("Expected stack of vectors or matrices, "
                                 "got shape {x.shape}")

        if len(y) == 1:
            return y[0]
        return y

    def log_prior(self, truths, params=None):
        if params is None:
            params = self.fit_parameters
        truths = self.select_params(truths, params)
        n_images = len(truths)
        result = np.ones(n_images)
        for i, pname in enumerate(params):
            x = truths[:,i]

            log = pname.startswith('log_')
            if log:
                # Network predicts log
                pname = pname[len('log_'):]
                # TODO: for log norm, could recover mu and sigma
                # to greatly simplify & stabilize this...
                x = np.exp(x)
            result *= self.prior_dists[pname].pdf(x)

        return result

    def logl(self, truths, preds, precs, params=None):
        """Return array of log likelihood ratios for several images

        Truth-independent terms (e.g. log det sigma) are omitted

        Args:
         - truths, preds: Truths and predictions, (n_images, n_params)
         - precs: Precision matrices, (n_images, n_params, n_params)
         - params: list of parameters to use. If
        """
        if params is None:
            params = self.fit_parameters
        n_params = len(params)
        truths, preds, precs = self.select_params(truths, preds, precs, params)

        single_image_call = (len(truths.shape) == 1)
        if single_image_call:
            truths = truths[None,...]
            preds = preds[None,...]
            precs = precs[None,...]
        n_images = truths.shape[0]
        assert truths.shape == preds.shape == (n_images, n_params)
        assert precs.shape == (n_images, n_params, n_params)

        diff = preds - truths
        log_posterior = -0.5 * diff[:,None,:] @ precs @ diff[:,:,None]

        log_l = log_posterior - self.log_prior(truths, params)

        if single_image_call:
            return log_l[0]
        return log_l


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
