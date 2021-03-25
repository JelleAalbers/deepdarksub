from dataclasses import dataclass

import astropy
import lenstronomy as ls
import numpy as np

import manada
import deepdarksub as dds
export, __all__ = dds.exporter()

ls.laconic()


@export
@dataclass
class LensMaker:
    """Utility for making single lensed images
    """

    def __init__(self, manada_config=None):
        self.manada_config = mc = dds.load_manada_config(manada_config)
        c = mc.config_dict

        # Load parameters we need from manada config
        self.n_pixels = mc.numpix
        self.pixel_width = c['detector']['parameters']['pixel_scale']
        self.psf_fwhm = c['psf']['parameters']['fwhm']
        self.z_lens = c['main_deflector']['parameters']['z_lens']
        self.z_source = c['source']['parameters']['z_source']
        self.subhalo_concentration = 15   # TODO: It's more complicated
        self.subhalo_truncation = 5       # TODO: It's more complicated
        self.smoothing_sigma = c['source']['parameters']['smoothing_sigma']

        self.image_length = self.pixel_width * self.n_pixels
        self.catalog = c['source']['class'](
            source_parameters=c['source']['parameters'],
            cosmology_parameters=c['cosmology']['parameters'])

    def subhalo_lenses(self, masses, positions):
        """Return list of (model string, config dict) for subhalo lenses
        :param masses: list of subhalo masses
        :param positions: list of (x,y) subhalo positions (arcsec, on lens plane)
        """
        # LensCosmo handles cosmology-dependent parameter conversions
        lens_cosmo = ls.LensCosmo(
            z_lens=self.z_lens,
            z_source=self.z_source,
            cosmo=astropy.cosmology.default_cosmology.get())

        lenses = []
        for m, (x, y) in zip(masses, positions):
            # Convert mass and concentration to lensing parameters
            # (return values flipped in docstring, fixed in March 2021 lenstronomy)
            r_scale, r_scale_bending = lens_cosmo.nfw_physical2angle(
                M=m, c=self.subhalo_concentration)
            print(r_scale, r_scale_bending)

            subhalo = ('TNFW', {
                # Scale radius
                'Rs': r_scale,
                # Observed bending angle at scale radius
                'alpha_Rs': r_scale_bending,
                # Truncation radius
                'r_trunc': self.subhalo_truncation * r_scale,
                'center_x': x, 'center_y': y})
            lenses.append(subhalo)
        return lenses

    def lensed_image(self, lenses=None, catalog_i=None):
        """Return numpy array describing lensed image
        :param lenses: list of (lens model name, lens kwargs)
        :param catalog_i: image index from COSMOS catalog
        """
        if lenses is None:
            # Do not lens
            lenses = [('SIS', dict(theta_E=0.))]
        if catalog_i is None:
            catalog_i = self.catalog.sample_indices(1)[0]

        lens_model_names, lens_kwargs = list(zip(*lenses))

        source_model_class, kwargs_source = self.catalog.draw_source(
            catalog_i=catalog_i,
            z_new=self.z_source)

        return ls.ImageModel(
                data_class=ls.ImageData(**ls.data_configure_simple(
                    numPix=self.n_pixels,
                    deltaPix=self.pixel_width)),
                psf_class=ls.Data.psf.PSF(
                    psf_type='GAUSSIAN',
                    fwhm=self.psf_fwhm),
                lens_model_class=ls.LensModel_(lens_model_names),
                source_model_class=ls.LightModel_(source_model_class),
            ).image(
                kwargs_lens=lens_kwargs,
                kwargs_source=kwargs_source)
