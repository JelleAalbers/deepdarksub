from functools import partial

import fastai.vision.all as fv
import numpy as np
import PIL
from scipy import stats
import torch

import deepdarksub as dds
export, __all__ = dds.exporter()


@export
class NumpyImage(fv.PILImage):
    """Floating-point image stored in an .npy file"""
    _tensor_cls = fv.TensorImageBW
    # noisemaker: dds.NoiseMaker

    @classmethod
    def create(cls, fn, **kwargs):
        data = np.load(fn)
        # data += cls.noisemaker.noise_on_load(fn, data)

        # Assume any overall noise mean is already subtracted,
        # so we don't have to do e.g.
        # data -= np.percentile(data, 10)
        # Subtracting min would shift masked pixels
        # & gives some images with very narrow color ranges :-(

        # Divide by max, or should we divide by some percentile?
        data /= data.max()

        data = data.clip(0, 1)

        return cls(PIL.Image.fromarray(data))


@export
def repeat_color(x):
    """Repeat content of x along the color dimension"""
    if len(x.shape) > 2:
        # This should be an image
        return torch.repeat_interleave(x, 3, dim=-3)
    return x



@export
class UncertaintyLoss(fv.nn.Module):
    """Custom loss for nets that output values and uncertainties"""

    def __init__(self, n_params, *args, **kwargs):
        self.n_params = n_params
        super().__init__(*args, **kwargs)

    def forward(self, x, y, **kwargs):
        # Split off the uncertainty terms.
        # Note y_unc is just a dummy and not used. (maybe we can remove it?)
        x_unc, y_unc = x[:, self.n_params:], y[:, self.n_params:]
        x, y = x[:, :self.n_params], y[:, :self.n_params]

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

        return loss.mean()


def _get_y(fn, meta, normalizer, add_uncertainty, fit_parameters):
    """Return list of desired outputs for image filename fn"""
    q = dds.meta_for_filename(meta, fn)

    # Get fit parameters
    y = [normalizer.norm(q[p], p) for p in fit_parameters]
    if add_uncertainty:
        # Add dummy outputs for uncertainties
        # (values do not matter, these do not enter the loss)
        y += [1] * len(fit_parameters)
    return y


@export
def data_block(
        meta, fit_parameters, data_dir,
        add_uncertainty=True,
        augment_rotation='free',
        rotation_pad_mode='zeros',
        do_repeat_color=False,
        **kwargs):
    """Return datablock for setting up learner"""
    n_params = len(fit_parameters)
    normalizer = dds.Normalizer(meta, fit_parameters)

    # Rotation augmentation makes fitting x, y, angles, etc. tricky!
    # (would have to figure out how to transform labels...)
    batch_tfms = []
    if augment_rotation == 'free':
        batch_tfms += [fv.Rotate(p=1., max_deg=180, pad_mode=rotation_pad_mode)]
    elif augment_rotation == 'right':
        # TODO: This doesn't actually randomly rotate along right angles,
        # just draws one batch of rotation angles to use.
        # If you change the batch size, you die here.
        batch_tfms += [
            fv.Rotate(p=1.,
                      draw=np.random.choice([0, 90, 180, 270], size=64).tolist())]

    # Without this, we get strange train/valid differences once we train the network
    # (e.g. 3 colors in train, 1 in val; or val loss stagnates very soon -> normalization diff?)
    # Maybe related: https://github.com/fastai/fastai/issues/3250
    batch_tfms += [fv.Normalize(mean=torch.tensor(0.), std=torch.tensor(1.))]

    return fv.DataBlock(
        blocks=(fv.TransformBlock(
                    type_tfms=NumpyImage.create,
                    batch_tfms=[repeat_color] if do_repeat_color else []),
                fv.RegressionBlock(n_out=2 * n_params if add_uncertainty else 2)),
        get_items=lambda _: tuple([data_dir / fn
                                   for fn in meta['filename'].values.tolist()]),
        get_y=partial(_get_y,
                      meta=meta,
                      normalizer=normalizer,
                      add_uncertainty=add_uncertainty,
                      fit_parameters=fit_parameters),
        splitter=fv.FuncSplitter(lambda fn: dds.meta_for_filename(meta, fn).is_val),
        batch_tfms=batch_tfms,
        **kwargs)
