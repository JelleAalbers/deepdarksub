"""NOT currently used!
"""

import lenstronomy as ls; ls.laconic()
import fastai.vision.all as fv
import numpy as np
import torch

import deepdarksub as dds
export, __all__ = dds.exporter()


@export
class NoiseMaker:

    def __init__(self, metadata, manada_config=None, noise_handling='leave'):
        """
        :param metadata: DataFrame of source images
        :param manada_config: manada config module
        :param noise_handling:
          - "leave": Training data already has noise. Leave it alone.
          - "load": Add noise during loading; always the same when loading
            the same image.
          - "augment": Add noise during training as data augmentation.
            That is: for training data, use different noise every time
                the same image is loaded. For validation data, always use
                the same noise when loading the same image.
                (Different images still have different noise.)
        """
        manada_config = dds.load_manada_config(manada_config)
        self.meta = metadata
        self.noise_handling = noise_handling

        det_pars = manada_config.config_dict['detector']['parameters']
        band = ls.SingleBand(**det_pars)
        self.readout_noise_scale = band.flux_noise(1)
        self.background_noise = band.background_noise

    def noise_on_load(self, fn, img):
        """Return noise to add to image"""
        if self.noise_handling == 'leave':
            return img
        image_index = dds.filename_to_index(fn)
        m = self.meta.loc[image_index]

        if (self.noise_handling == 'load'
                or self.noise_handling == 'augment' and m.is_val):
            # Apply fixed noise to the image
            rng = np.random.default_rng(seed=image_index)
            return img + rng.normal(scale=self.noise_sigma(img))

    def noise_sigma(self, img):
        return (self.background_noise**2
                + self.readout_noise_scale**2 * img.clip(0, None)
                )**0.5

    def noise_transforms(self):
        """Return list of fastai transforms to add noise in augmentation.
        The list is empty unless noise_handling is set to 'augment'.
        """
        if self.noise_handling == 'augment':
            return [AddNoise(noisemaker=self, p=1)]
        else:
            return []


class AddNoise(fv.RandTransform):
    def __init__(self, noisemaker, *args, **kwargs):
        self.noisemaker = noisemaker
        super().__init__(*args, **kwargs)

    def encodes(self, x: fv.TensorImage):
        x = x + torch.normal(mean=x * 0,
                             std=self.noisemaker.noise_sigma(x))
        return x.clip(0, 1)

# # Test we're doing flux scaling just as lenstronomy
# img = np.load(data_dir / meta.iloc[0]['filename'])
# np.testing.assert_almost_equal(band.flux_noise(img )* *2,
#                                readout_noise_scal e* *2 * img.clip(0, None))
# del img