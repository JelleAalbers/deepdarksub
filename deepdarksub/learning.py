import contextlib
from immutabledict import immutabledict
from pathlib import Path
from functools import partial
import tempfile

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
def image_to_input(img, device='cpu'):
    """Return neural net input given one or more images
        (useful for feature attribution)

    Args:
     - img: numpy array or image filename

    Returns: (1, 1, image_dim1, image_dim2) tensor
        (axis 0 is the batch dimension, axis 1 color)
    """
    if isinstance(img, (tuple, list)):
        # Concatenate along batch dimension
        return torch.cat(
            tuple([image_to_input(_img) for _img in img]),
            dim=0).to(device)
    if isinstance(img, np.ndarray):
        # Use a temporary file, to ensure exactly the same initialization
        # code is run.
        with tempfile.NamedTemporaryFile() as tempf:
            np.save(tempf, img)
            img = dds.NumpyImage.create(Path(tempf.name))
    else:
        img = dds.NumpyImage.create(Path(img))
    img = torch.tensor(np.array(img), device=device)
    # Add batch dimension (1) and channel dimension (1 since monochrome)
    img = img[None, None, :,:]
    return img


@export
def repeat_color(x):
    """Repeat content of x along the color dimension"""
    if len(x.shape) > 2:
        # This should be an image
        return torch.repeat_interleave(x, 3, dim=-3)
    return x


class MyLabel(fv.TensorBase):
    pass


def _rotate_inplace(o, i1, i2, cos, sin):
    o[:, i1], o[:, i2] = (
        cos * o[:, i1] - sin * o[:, i2],
        sin * o[:, i1] + cos * o[:, i2])


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
        fp = list(fit_parameters)[0]
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

        # Use fastai's type-aware monkeypatching to register how
        # rotations work. Mostly harmless... Just don't try to do more than one
        # inference at a time with different parameters,
        # and don't expect unpickling to work nicely....

        _index_of = immutabledict({
            x: list(fit_parameters).index(
                'main_deflector_parameters_' + x)
            for x in (
                'e1', 'e2',
                'gamma1', 'gamma2',
                'center_x', 'center_y')})

        @fv.Rotate
        def encodes(self, o: MyLabel, _i=_index_of):
            assert self.mat.shape[1], self.mat.shape[2] == (2, 3)

            # Checked these are correct with draw=10 in Rotate
            cos_q, sin_q = self.mat[:, 0, 0], self.mat[:, 0, 1]

            # Ellipticity and shear rotate at double speed
            sin_2q = 2 * sin_q * cos_q
            cos_2q = 2 * cos_q**2 - 1

            _rotate_inplace(o, _i['e1'], _i['e2'], cos_2q, sin_2q)
            _rotate_inplace(o, _i['gamma1'], _i['gamma2'], cos_2q, sin_2q)
            _rotate_inplace(o, _i['center_x'], _i['center_y'], cos_q, sin_q)

            return o

        get_y = partial(_get_y_regression,
                        meta=meta,
                        normalizer=normalizer,
                        fit_parameters=fit_parameters,
                        label_class=MyLabel)

    # Rotation augmentation makes fitting x, y, angles, etc. tricky!
    # (would have to figure out how to transform labels...)
    batch_tfms = []
    if augment_rotation == 'free':
        batch_tfms += [fv.Rotate(p=1.,
                                 max_deg=180,
                                 pad_mode=rotation_pad_mode)]
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


def _get_y_regression(fn, *, meta, normalizer, fit_parameters, label_class):
    """Return list of desired outputs for image filename fn"""
    q = dds.meta_for_filename(meta, fn)

    y = [normalizer.norm(q[p], p) for p in fit_parameters]
    y += [q.training_weight]
    return label_class(y, fit_parameters=fit_parameters)


def _get_y_classification(fn, *, meta):
    """Returns desired output for image filename fn"""
    return dds.meta_for_filename(meta, fn).class_index


def _get_y_segmentation(fn, *, mask_dir):
    return Path(mask_dir) / (fn.stem + '.png')


@export
class TestTimeDropout(fv.Callback):
    """`Callback` that activates Dropout layers at inference time if _active
    Adapted from https://forums.fast.ai/t/mc-dropout-in-fastai-v2/68247
    """
    _active = False

    def dropout_modules(self):
        for mod in self.model.modules():
            if 'dropout' in mod.__class__.__name__.lower():
                yield mod

    @contextlib.contextmanager
    def active(self, p_mult=1.):
        if not p_mult:
            # Do not activate dropout
            yield
            return
        if p_mult is True:
            p_mult == 1.

        _orig_ps = {mod: mod.p
                    for mod in self.dropout_modules()}
        try:
            self._active = True
            for mod in self.dropout_modules():
                mod.p *= p_mult
            yield
        finally:
            self._active = False
            for mod in self.dropout_modules():
                mod.p = _orig_ps[mod]

    # Why not set train / val as part of active?
    # And if that works, why make a callback at all?

    def before_batch(self):
        if not self._active:
            return
        for mod in self.dropout_modules():
            mod.train()

    def after_batch(self):
        if self._active:
            return
        for mod in self.dropout_modules():
            mod.eval()
