import fastai.vision.all as fv
import numpy as np
import torch

import deepdarksub as dds
export, __all__ = dds.exporter()


@export
def loss_for(n_params, uncertainty, soft_loss_max=1000):
    if not uncertainty:
        return WeightedLoss(n_params)
    elif uncertainty == 'diagonal':
        return UncertaintyLoss(n_params)
    elif uncertainty == 'correlated':
        return CorrelatedUncertaintyLoss(n_params, soft_loss_max=soft_loss_max)
    else:
        raise ValueError(f"Uncertainty {uncertainty} not recognized")


@export
class WeightedLoss(fv.nn.Module):

    def __init__(self, n_params, *args, **kwargs):
        self.n_params = n_params
        super().__init__(*args, **kwargs)

    def forward(self, x, y):
        assert y.shape == (x.shape[0], self.n_params + 1)
        y, weight = y[:, :self.n_params], y[:, self.n_params]
        return (weight * self.loss(x, y)).mean()

    def loss(self, x, y):
        assert x.shape == y.shape
        # Mean absolute error
        # return torch.mean(torch.abs(x - y), dim=1)
        # RMSE
        return torch.mean((x - y)**2, dim=1)**0.5


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
        loss = (torch.abs(x - y) / x_unc).mean(axis=1)

        # Part 2: uncertainty
        #    Not sure how to weight these
        #    0.5, 1: seem OK both
        #    0.2: errors just stay at 1
        loss += x_unc.mean(axis=1)

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
        x_p, L = dds.x_to_xp_L(x, self.n_params)
        assert x_p.shape == y.shape

        # Loss part 1: Squared Mahalanobis distance
        delta = x_p - y
        q = torch.einsum('bi,bij->bj', delta, L)
        loss1 = torch.einsum('bi,bi->b', q, q)

        # Part 2: penalty for large uncertainties/covariances
        loss2 = - 2 * torch.diagonal(torch.log(L), dim1=-2, dim2=-1).sum(-1)

        loss = loss1 + loss2

        # Clip the loss (gently, so we can still differentiate)
        # to avoid insanely high values
        # (insanely low values are welcome :-)
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
def matrix_diag_batched(diagonal):
    """Convert batch of vectors to batch of diagonal matrices"""
    # From https://github.com/pytorch/pytorch/issues/5198#issuecomment-425069863
    N = diagonal.shape[-1]
    shape = diagonal.shape[:-1] + (N, N)
    device, dtype = diagonal.device, diagonal.dtype
    result = torch.zeros(shape, dtype=dtype, device=device)
    indices = torch.arange(result.numel(), device=device).reshape(shape)
    indices = indices.diagonal(dim1=-2, dim2=-1)
    result.view(-1)[indices] = diagonal
    return result


@export
def x_to_xp_L(x, n):
    """Return (x, L) given net output x
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

    # Find indices for L matrix assignment
    # See https://discuss.pytorch.org/t/upper-triangular-matrix-vectorization/7040
    with torch.no_grad():
        # Indices of off-diagonal lower triangular elements
        indices = (
                1 == 1 - torch.triu(torch.ones(n, n))
            ).expand(n_batch, n, n).to(x.device)
        L = torch.zeros(n_batch, n, n).to(x.device)

    L[indices] = x_nondiag.reshape(-1)
    L = L + matrix_diag_batched(torch.exp(x_diag))
    return x_p, L


@export
def L_to_cov(L):
    if isinstance(L, torch.Tensor):
        L = L.numpy()
    inverse_cov = L @ np.transpose(L, (0, 2, 1))
    return np.stack([np.linalg.inv(x) for x in inverse_cov],
                    axis=0)
