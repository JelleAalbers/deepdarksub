import numpy as np
import torch
import deepdarksub as dds


def loss_numpy(x, y):
    n = 2
    x_p, x_diag, x_nondiag = x[:n], x[n:2 * n], x[2 * n:]
    x_diag_exp = np.exp(x_diag)
    y_p = y[:n]
    L = np.array([[x_diag_exp[0], 0],
                  [x_nondiag[0], x_diag_exp[1]]])
    delta = x_p - y_p
    loss1 = (delta @ L) @ (delta @ L)
    loss2 = -2 * x_diag.sum()
    print("Numpy loss: ", loss1, loss2)
    return loss1 + loss2


def test_uncertainty_loss():
    # Random 2-parameter inputs
    # TODO: randomize with hypothesis!
    x, y = np.random.rand(5), np.random.rand(5)

    # Reference computation in numpy
    loss = loss_numpy(x, y)

    # Convert to pytorch - do batch size = 1 first
    xt, yt = torch.Tensor(x).expand(1, 5), torch.Tensor(y).expand(1, 5)

    # Test batch size = 1
    np.testing.assert_almost_equal(
        dds.CorrelatedUncertaintyLoss(2)(xt, yt),
        loss,
        decimal=3)

    # Test with batch size = 42. Loss should be averaged over batch.
    xt, yt = xt * torch.ones((42, 5)), yt * torch.ones((42, 5))
    np.testing.assert_almost_equal(
        dds.CorrelatedUncertaintyLoss(2)(xt, yt),
        loss,
        decimal=3)

    # Test against alternate form of loss
    # or equivalently, test decoding into cov matrix
    _, L = dds.x_to_xp_L(torch.Tensor(x[None,:]), 2)
    cov = dds.L_to_cov(L)[0]
    inv_cov = np.linalg.inv(cov)
    delta = x[:2] - y[:2]
    loss_alt_1 = (delta @ inv_cov @ delta).squeeze()
    loss_alt_2 = np.log(np.linalg.det(cov))
    print("Alternate loss: ", loss_alt_1, loss_alt_2)
    np.testing.assert_almost_equal(
        loss,
        loss_alt_1 + loss_alt_2)
