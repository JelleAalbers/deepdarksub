import contextlib
from pathlib import Path
from functools import partial

import fastai.vision.all as fv
import numpy as np
import PIL
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

        # Log and normalize
        # vmax = np.max(data)
        # vmin = 1e-3 * vmax
        # data = np.log(data.clip(vmin, vmax))
        # vmin, vmax = np.log(vmin), np.log(vmax)
        # data = (data - vmin) / (vmax - vmin)

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
def data_block(
        meta, fit_parameters, data_dir,
        uncertainty,
        class_thresholds=None,
        mask_dir=None,
        augment_rotation='free',
        rotation_pad_mode='zeros',
        do_repeat_color=False,
        **kwargs):
    """Return datablock for setting up learner"""
    n_params = len(fit_parameters)

    if class_thresholds is not None:
        # Classification: find bounds
        # TODO: bit sloppy to put these in global metadata...
        assert len(fit_parameters) == 1
        fp = list(fit_parameters.keys())[0]
        meta['class_index'] = np.searchsorted(
            class_thresholds,
            meta[fp].values)
        out_block = fv.CategoryBlock()
        get_y = partial(_get_y_classification, meta=meta)

    elif mask_dir is not None:
        # Segmentation
        with open(mask_dir / 'codes.txt') as f:
            codes = f.read().split(', ')
        out_block = fv.MaskBlock(codes=codes)
        get_y = partial(_get_y_segmentation, mask_dir=mask_dir)

    else:
        # Regression
        normalizer = dds.Normalizer(meta, fit_parameters)
        out_block = fv.RegressionBlock(
            n_out=dds.n_out(n_params, uncertainty=uncertainty))
        get_y = partial(_get_y_regression,
                        meta=meta,
                        normalizer=normalizer,
                        fit_parameters=fit_parameters)

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
                out_block),
        get_items=lambda _: tuple([data_dir / fn
                                   for fn in meta['filename'].values.tolist()]),
        get_y=get_y,
        splitter=fv.FuncSplitter(lambda fn: dds.meta_for_filename(meta, fn).is_val),
        batch_tfms=batch_tfms,
        **kwargs)


def _get_y_regression(fn, *, meta, normalizer, fit_parameters):
    """Return list of desired outputs for image filename fn"""
    q = dds.meta_for_filename(meta, fn)
    y = [normalizer.norm(q[p], p) for p in fit_parameters]
    y += [q.training_weight]
    return y


def _get_y_classification(fn, *, meta):
    """Returns desired output for image filename fn"""
    return dds.meta_for_filename(meta, fn).class_index


def _get_y_segmentation(fn, *, mask_dir):
    return Path(mask_dir) / (fn.stem + '.png')


@export
def predict_many(learn, normalizer, filenames,
                 uncertainty=False,
                 progress=True,
                 **kwargs):
    if isinstance(filenames, (str, Path)):
        filenames = [filenames]
    dl = learn.dls.test_dl(filenames, **kwargs)
    with (contextlib.nullcontext() if progress else learn.no_bar()):
        preds = learn.get_preds(dl=dl)[0]
    return normalizer.decode(preds, uncertainty=uncertainty)
