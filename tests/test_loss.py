import numpy as np
from scipy import stats
import torch

import deepdarksub as dds


def loss_numpy(x, y, truncate_final=False):
    n = 2
    x_p, x_diag, x_nondiag = x[:n], x[n:2 * n], x[2 * n:]
    x_diag_exp = np.exp(x_diag)
    y_p = y[:n]
    L = np.array([[x_diag_exp[0], 0],
                  [x_nondiag[0], x_diag_exp[1]]])
    delta = x_p - y_p
    # Same as
    # loss1 = (L.T @ delta).T @ (L.T @ delta)
    loss1 = (delta @ L) @ (delta @ L)
    loss2 = -2 * x_diag.sum()

    if truncate_final:
        std, _ = dds.cov_to_std(dds.L_to_cov(L))
        f_above = stats.norm(loc=x_p[-1], scale=std[-1]).sf(0)
        loss2 += 2 * np.log(f_above)
    print("Numpy loss: ", loss1, loss2)

    return loss1 + loss2


def test_uncertainty_loss():
    # Random 2-parameter input, training weight of 1
    # TODO: randomize with hypothesis!
    x, y = np.random.rand(5), np.random.rand(2)
    fit_parameters = ['foo', 'bar']
    y = np.concatenate([y, [1]])

    # Reference computation in numpy
    loss = loss_numpy(x, y)

    # Convert to pytorch - do batch size = 1 first
    xt = torch.Tensor(x).expand(1, len(x))
    yt = torch.Tensor(y).expand(1, len(y))

    # Test batch size = 1
    np.testing.assert_almost_equal(
        dds.loss_for(fit_parameters, 'correlated')(xt, yt),
        loss,
        decimal=3)

    # Test with truncation term
    np.testing.assert_almost_equal(
        dds.loss_for(fit_parameters, 'correlated', truncate_final=True)(xt, yt),
        loss_numpy(x, y, truncate_final=True),
        decimal=3)

    # Test with batch size = 42. Loss should be averaged over batch.
    xt, yt = xt * torch.ones((42, 5)), yt * torch.ones((42, 3))
    np.testing.assert_almost_equal(
        dds.loss_for(fit_parameters, 'correlated')(xt, yt),
        loss,
        decimal=3)

    # Test against alternate form of loss
    # or equivalently, test decoding into cov matrix
    _, L, _ = dds.x_to_xp_L(torch.Tensor(x[None,:]), 2)
    cov = dds.L_to_cov(L)[0]
    inv_cov = np.linalg.inv(cov)
    delta = x[:2] - y[:2]
    loss_alt_1 = (delta @ inv_cov @ delta).squeeze()
    loss_alt_2 = np.log(np.linalg.det(cov))
    print("Alternate loss: ", loss_alt_1, loss_alt_2)
    np.testing.assert_almost_equal(
        loss,
        loss_alt_1 + loss_alt_2)
