import fastai.vision.all as fv
import numpy as np
import torch

import deepdarksub as dds
export, __all__ = dds.exporter()


@export
def loss_for(fit_parameters, uncertainty, soft_loss_max=1000, 
             parameter_weights=None,
             weight_factors=None):
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
            parameter_weights)
    elif uncertainty == 'diagonal':
        return UncertaintyLoss(
            n_params, 
            parameter_weights)
    elif uncertainty == 'correlated':
        return CorrelatedUncertaintyLoss(
            n_params,
            parameter_weights,   # Inert
            soft_loss_max=soft_loss_max)
    else:
        raise ValueError(f"Uncertainty {uncertainty} not recognized")


@export
class WeightedLoss(fv.nn.Module):

    def __init__(self, n_params, parameter_weights, *args, **kwargs):
        assert len(parameter_weights) == n_params
        # Weights must sum to 1.
        parameter_weights = parameter_weights / parameter_weights.sum()

        self.n_params = n_params
        self.parameter_weights = parameter_weights
        super().__init__(*args, **kwargs)

    def forward(self, x, y, reduction='mean'):
        assert y.shape == (x.shape[0], self.n_params + 1)
        y, weight = y[:, :self.n_params], y[:, self.n_params]
        loss = weight * self.loss(x, y)
        if reduction == 'mean':
            return loss.mean()
        assert reduction == 'none'
        return loss

    def loss(self, x, y):
        assert x.shape == y.shape
        # Mean absolute error
        # return torch.mean(torch.abs(x - y), dim=1)
        # RMSE
        return torch.sum((x - y)**2 * self.parameter_weights[None,:],
                         dim=1)**0.5


@export
class UncertaintyLoss(WeightedLoss):
    """Custom loss for nets that output values and an estimate of
    the uncertainty per variable
    """

    def loss(self, x, y):
        # Split off the actual parameters from uncertainties
        x, x_unc = x[:, :self.n_params], x[:, self.n_params:]
        assert x.shape == y.shape == x_unc.shape

        # Let neural net predict the log2 of the uncertainty.
        # (Maybe bad, but you have to give some meaning to negative values)
        x_unc = 2 ** x_unc

        # Part 1: abs error / uncertainty
        loss = torch.sum(
            ((x - y) / x_unc)**2
                * self.parameter_weights[None,:],
            dim=1)

        # Part 2: uncertainty
        #    Not sure how to weight these
        #    0.5, 1: seem OK both
        #    0.2: errors just stay at 1
        loss += 2 * torch.sum(
            torch.log(x_unc) * self.parameter_weights[None,:],
            dim=1)

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

        # Part 2: log determinant of the covariance matrix
        # Note that for many-parameter fits, these determinants get tiny.
        # log-ing diag(L) would be less numerically stable,
        # as L is built via exp.
        loss2 = - 2 * log_diag_L.sum(-1)

        loss = loss1 + loss2

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
    """Return (x, L, log(diag(L))) given net output x
    :param x: Neural net output, (n_batch, n_outputs)
    :param n: Number of variables

    Net predicts (
        point estimates,
        ln of diagonal entries of L,
        off-diagonal entries of L
    )
    """
    if not isinstance(x, torch.Tensor):
        x = torch.Tensor(x)
    n_batch, n_out = x.shape
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
    inverse_cov = L @ np.transpose(L, (0, 2, 1))
    return np.stack([np.linalg.inv(x) for x in inverse_cov],
                    axis=0)
