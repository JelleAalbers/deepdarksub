import fastai.vision.all as fv
import numpy as np
import torch

import deepdarksub as dds
export, __all__ = dds.exporter()


@export
def loss_for(fit_parameters, uncertainty, soft_loss_max=1000,
             truncate_final_to=None,
             parameter_weights=None):
    n_params = len(fit_parameters)

    if parameter_weights is None:
        parameter_weights = dict()
    elif uncertainty == 'correlated':
        raise NotImplementedError(
            "Correlated loss does not support "
            "parameter weighting (yet)")

    if isinstance(parameter_weights, dict):
        parameter_weights = [
            parameter_weights.get(p, 1)
            for p in fit_parameters]
    parameter_weights = torch.Tensor(parameter_weights).cuda()

    if not uncertainty:
        return WeightedLoss(
            n_params,
            parameter_weights,
            truncate_final_to=truncate_final_to)
    elif uncertainty == 'diagonal':
        return UncertaintyLoss(
            n_params,
            parameter_weights,
            truncate_final_to=truncate_final_to)
    elif uncertainty == 'correlated':
        return CorrelatedUncertaintyLoss(
            n_params,
            parameter_weights,   # Inert
            truncate_final_to=truncate_final_to,
            soft_loss_max=soft_loss_max)
    else:
        raise ValueError(f"Uncertainty {uncertainty} not recognized")


@export
class WeightedLoss(fv.nn.Module):

    def __init__(self, n_params, parameter_weights, *args,
                 truncate_final_to=None,
                 **kwargs):
        self.n_params = n_params
        assert len(parameter_weights) == n_params
        # Force weights to sum to one
        parameter_weights = parameter_weights / parameter_weights.sum()
        self.parameter_weights = parameter_weights
        self.truncate_final_to = truncate_final_to
        super().__init__(*args, **kwargs)

    def forward(self, x, y, reduction='mean'):
        assert y.shape == (x.shape[0], self.n_params + 1)
        y, weight = y[:, :self.n_params], y[:, self.n_params]
        loss = weight * self.loss(x, y)
        if reduction == 'mean':
            return loss.mean()
        assert reduction == 'none'
        return loss

    def truncation_term(self, mean, std=1):
        """Return term to be added to -2 log L loss for truncating the
        estimated posterior of a fit_parameter with to non-negative physical
        values.
        
        If the posterior is >90% out of range, only part of the normalization will 
        be compensated. This helps avoid
          (a) NAN NAN NAN when starting training
          (b) ridiculously low predictions

        Args:
         - mean: Predicted mean
         - std: Predicted std
        """
        if self.truncate_final_to is None:
            return 0.
        # 1 - Normal(mean,std).CDF(0)
        f_above = 1 - 0.5 * (1 + torch.erf((self.truncate_final_to - mean)
                                           /(std * 2**0.5)))
        # print(f"f_above range: {f_above.min():.4f}, {f_above.max():.4f}")
        f_above = f_above.clamp(0.1, 1)

        # -2 log (1/f_above) = 2 log f_above
        return 2 * torch.log(f_above)

    def loss(self, x, y):
        assert x.shape == y.shape
        return (
            torch.sum((x - y)**2 * self.parameter_weights[None,:],
                      dim=1)
            + self.truncation_term(mean=x[:,-1]))


@export
class UncertaintyLoss(WeightedLoss):
    """Custom loss for nets that output values and an estimate of
    the uncertainty per variable
    """

    def loss(self, x, y):
        # Split off the actual parameters from uncertainties
        x, x_unc = x[:, :self.n_params], x[:, self.n_params:]
        assert x.shape == y.shape == x_unc.shape

        # Let neural net predict the log2 of the (std) uncertainty.
        # (Maybe bad, but you have to give some meaning to negative values)
        x_unc = 2 ** x_unc

        # Part 1: squared Mahalanobis distance (assuming no correlations)
        loss = torch.sum(
            ((x - y) / x_unc)**2
                * self.parameter_weights[None,:],
            dim=1)

        # Part 2: -2 log(det_of_cov**-0.5) = sum log x_unc (no 2!)
        loss += torch.sum(
            torch.log(x_unc) * self.parameter_weights[None,:],
            dim=1)

        loss += self.truncation_term(mean=x[:,-1], std=x_unc[:,-1])

        return loss


@export
class CorrelatedUncertaintyLoss(WeightedLoss):
    """Custom loss for nets that output values and an estimate of
    the Cholesky-decomposed inverse covariance matrix L
    See https://arxiv.org/pdf/1802.07079.pdf (not the sparse part)
    """

    def __init__(self, *args, soft_loss_max=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.soft_loss_max = soft_loss_max

    def loss(self, x, y):
        x_p, L, log_diag_L = dds.x_to_xp_L(x, self.n_params)
        assert x_p.shape == y.shape

        # Loss part 1: Squared Mahalanobis distance
        delta = x_p - y
        q = torch.einsum('bi,bij->bj', delta, L)
        loss1 = torch.einsum('bi,bi->b', q, q)

        # Part 2: -2 log(det_of_cov**-0.5) = log det_of_cov = -2 sum log diag(L)
        # (since cov = (L L^T)^-1 and L is lower triangular)
        # Note that for many-parameter fits, these determinants get tiny.
        # log-ing diag(L) would be less numerically stable,
        # as L is built via exp.
        loss2 = - 2 * log_diag_L.sum(-1)

        loss = loss1 + loss2
        # 1/L[-1,-1] equals the standard deviation on the final parameter.
        # See https://math.stackexchange.com/a/3211183, or just try it out:
        #   L = np.tril(np.random.rand(13,13))
        #   L[-1, -1]**-1, dds.cov_to_std(dds.L_to_cov(L))[0][-1]
        loss += self.truncation_term(mean=x_p[:, -1], std=L[:, -1, -1]**-1)

        # Clip the loss to avoid insanely high values, common at initialization.
        # Do it gently, so we can still learn from saturating examples.
        loss = dds.soft_clip_max(loss, self.soft_loss_max)

        return loss


@export
def n_out(n, uncertainty):
    if not uncertainty:
        return n
    elif uncertainty == 'diagonal':
        return 2 * n
    elif uncertainty == 'correlated':
        return n + int(n * (n+1)/2)
    else:
        raise ValueError(f"{uncertainty} not a recognized uncertainty")


@export
def x_to_xp_L(x, n):
    """Return (x, L, ln(diag(L))) given net output x
    :param x: Neural net output, (n_batch, n_outputs)
    :param n: Number of variables

    Note that the net predicts: (
        point estimates,
        ln of diagonal entries of L,
        off-diagonal entries of L (flattened)
    )
    """
    if not isinstance(x, torch.Tensor):
        x = torch.Tensor(x)
    _, n_out = x.shape
    assert n_out == dds.n_out(n, 'correlated'), "Wrong number of outputs"

    # Split off the covariance terms from x
    x_p, x_diag, x_nondiag = x[:, :n], x[:, n:2 * n], x[:, 2 * n:]

    # Create diagonal matrices
    L = torch.diag_embed(torch.exp(x_diag))

    # Get indices of elements below main diagonal
    row_indices, col_indices = torch.tril_indices(n, n, -1)

    # Add off-diagonal entries in-place
    L[:, row_indices, col_indices] += x_nondiag

    return x_p, L, x_diag


@export
def L_to_cov(L):
    if isinstance(L, torch.Tensor):
        L = L.numpy()
    single_matrix = len(L.shape) == 2
    if single_matrix:
        # Add extra axis, remove at end
        L = L.reshape(1, len(L), len(L))

    inverse_cov = L @ np.transpose(L, (0, 2, 1))
    cov = np.stack([np.linalg.inv(x) for x in inverse_cov],
                   axis=0)

    if single_matrix:
        return cov[0]
    return cov
