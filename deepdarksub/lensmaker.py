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
    """Utility for making single lensing images
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
        self.catalog = manada.Sources.cosmos.COSMOSCatalog(
            source_parameters=dict(
                cosmos_folder=mc.cosmos_folder,
                smoothing_sigma=self.smoothing_sigma,
                # We're NOT using manada's galaxy sampling here
                minimum_size_in_pixels=float('nan'),
                min_apparent_mag=float('nan'),
                random_rotation=0,
                min_flux_radius=0,
                max_z=float('nan')),
            cosmology_parameters='planck18')

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

    def lensed_image(self, lenses, catalog_i):
        """Return numpy array describing lensed image
        :param lenses: list of (lens model name, lens kwargs)
        :param catalog_i: image index from COSMOS catalog
        """
        lens_model_names, lens_kwargs = list(zip(*lenses))
        return ls.ImageModel(
                data_class=ls.ImageData(**ls.data_configure_simple(
                    numPix=self.n_pixels,
                    deltaPix=self.pixel_width)),
                psf_class=ls.Data.psf.PSF(
                    psf_type='GAUSSIAN',
                    fwhm=self.psf_fwhm),
                lens_model_class=ls.LensModel_(lens_model_names),
                source_model_class=ls.LightModel_(['INTERPOL']),
            ).image(
                kwargs_lens=lens_kwargs,
                kwargs_source=self.catalog.draw_source(
                    catalog_i=catalog_i,
                    z_new=self.z_source)[1])

    # OLD!
    def select_galaxies(self):
        c = self.catalog.catalog
        # Sercic fit succeeded
        # From readme:
        #   FIT_STATUS: An array of 5 numbers indicating the status of the fit.  The first of those numbers is
        #   the status for BULGEFIT, and the last of those 5 numbers is the status for SERSICFIT.  A status of 0
        #   or 5 indicates failure.
        fit_ok = ~np.in1d(c['fit_status'][:, -1], [0, 5]) & c['viable_sersic']
        fit_ok.mean()

        # Size in 128 - 64 pixels
        size_min = np.minimum(c['size_x'], c['size_y'])
        size_max = np.maximum(c['size_x'], c['size_y'])
        size_ok = (64 <= size_min) & (size_max <= 128)

        # Remove photoz outliers
        photoz_ok = c['zphot'] < 1.8

        # Remove faint galaxies
        mag_ok = c['mag_auto'] < 22

        all_ok = (fit_ok & size_ok & photoz_ok & mag_ok).astype(np.bool_)
        print(f'Fit OK: {fit_ok.mean():.5f}\n'
              f'Size OK: {size_ok.mean():.5f}\n'
              # f'Signal/noise reasonable: {noise_ok.mean():.5f}\n'
              f'Photoz OK: {photoz_ok.mean():.5f}\n'
              f'Magnitude OK: {mag_ok.mean():.5f}\n'
              f'Select: {all_ok.sum()} images out of {len(all_ok)} ({all_ok.mean():.4f})')

        return np.where(all_ok)[0]
