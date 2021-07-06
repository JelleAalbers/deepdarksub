import lenstronomy as ls
from manada.Utils.cosmology_utils import get_cosmology
import numpy as np

import deepdarksub as dds
export, __all__ = dds.exporter()

ls.laconic()


@export
class LensMaker:
    """Utility for making single lensed images
    """

    def __init__(self, manada_config=None):
        self.manada_config = mc = dds.load_manada_config(manada_config)
        c = self.config_dict = mc.config_dict

        # Image dimensions
        self.n_pixels = mc.numpix
        self.pixel_width = c['detector']['parameters']['pixel_scale']
        self.image_length = self.pixel_width * self.n_pixels

        # Redshifts
        self.z_lens = c['main_deflector']['parameters']['z_lens']
        self.z_source = c['source']['parameters']['z_source']

        # Source catalog
        self.catalog = c['source']['class'](
            source_parameters=c['source']['parameters'],
            cosmology_parameters=c['cosmology']['parameters'])

        # Noise maker
        self.single_band = ls.SingleBand(**c['detector']['parameters'])

        self.astropy_cosmology = get_cosmology(
            self.config_dict['cosmology']['parameters']).toAstropy()

    def manada_main_deflector(self, **main_deflector_parameters):
        """Return list of main deflector lenses (model, kwarg pairs)"""
        c = self.config_dict
        main_deflector_parameters = {
            **dds.take_config_medians(c['main_deflector']['parameters']),
            **main_deflector_parameters}
        models = c['main_deflector']['models']
        return self._to_lens_list(
            models,
            [{k: v
             for k, v in main_deflector_parameters.items()
             if k in ls.ProfileListBase._import_class(model, None).param_names}
             for model in models])

    def _common_substructure_kwargs(self, main_deflector_parameters):
        c = self.config_dict
        return dict(
            main_deflector_parameters={
                **dds.take_config_medians(c['main_deflector']['parameters']),
                **main_deflector_parameters},
            source_parameters=c['source']['parameters'],
            cosmology_parameters=c['cosmology']['parameters'])

    def _to_lens_list(self, model_list, kwarg_list, zs=None):
        """Convert lists of (models, kwargs, optionally zs), i.e. what
        lenstronomy expects, to a list of (lens, kwargs) tuples
        """
        if zs is not None:
            for i, z in enumerate(zs):
                kwarg_list[i]['z'] = z
        return list(zip(model_list, kwarg_list))

    def _from_lens_list(self, lens_list):
        """Convert a list of (lens, kwargs) tuples to lists of
        (models, kwargs, zs).

        zs will be None if the lenses are all at z_lens
        """
        model_list, kwargs_list = list(zip(*lens_list))
        # Extract the 'z' I smuggled into the kwargs list
        zs = [kwargs.get('z', self.z_lens) for kwargs in kwargs_list]
        kwargs_list = [{k: v for k, v in kwargs.items() if k != 'z'}
                       for kwargs in kwargs_list]
        if all([z != self.z_lens for z in zs]):
            zs = None
        return model_list, kwargs_list, zs

    def manada_los(self,
                   delta_los=1.,
                   mode='subtract_average',
                   **main_deflector_parameters):
        """Return list of line-of-sight lenses.

        Args:
            delta_los: Scaling of los subhalo count
            mode:
                'subtract_average': draw subhalos, subtract average convergence
                'halos': draw subhalos
                'average_only': only subtract average convergence,
                    don't draw any subhalos
            **kwargs: main deflector parameters
        """
        c = self.config_dict
        los_maker = c['los']['class'](
            los_parameters={**c['los']['parameters'],
                            **dict(delta_los=delta_los)},
			**self._common_substructure_kwargs(main_deflector_parameters))
        result = []
        if mode in ['halos', 'subtract_average']:
            result.append(los_maker.draw_los())
        if mode in ['average_only', 'subtract_average']:
            # TODO: ask Sebastian about * 2 here
            result.append(los_maker.calculate_average_alpha(self.n_pixels * 2))
        return sum([self._to_lens_list(*x) for x in result], [])

    def manada_subhalos(self,
                        sigma_sub=0.1,
                        **main_deflector_parameters):
        """Return list of subhalo lenses (model, kwargs pairs), metadata dict"""
        c = self.config_dict
        subhalo_maker = c['subhalo']['class'](
			subhalo_parameters={**c['subhalo']['parameters'],
                                'sigma_sub': sigma_sub},
            **self._common_substructure_kwargs(main_deflector_parameters))

        # Steal some extra metadata from manada...
        spy_reports = dict()
        def spy_upon(object, method_name, alias=None):
            if alias is None:
                alias = method_name
            orig_f = getattr(object, method_name)
            def spied_f(*args, **kwargs):
                result = orig_f(*args, **kwargs)
                spy_reports[alias] = result
                return result
            setattr(object, method_name, spied_f)
        spy_upon(subhalo_maker, 'draw_nfw_masses', 'mass')
        spy_upon(subhalo_maker, 'mass_concentration', 'concentration')

        sub_model_list, sub_kwargs_list, _ = subhalo_maker.draw_subhalos()
        return self._to_lens_list(sub_model_list, sub_kwargs_list), spy_reports

    def simple_subhalos(self, masses, positions, concentration=15, truncation=5):
        """Return list of (model string, config dict) for subhalo lenses
        :param masses: list of subhalo masses
        :param positions: list of (x,y) subhalo positions (arcsec, on lens plane)

        Concentration=15, truncation=5 defaults come from
            https://arxiv.org/pdf/2009.06639.pdf.
        """
        # LensCosmo handles cosmology-dependent parameter conversions
        lens_cosmo = ls.LensCosmo(
            z_lens=self.z_lens,
            z_source=self.z_source,
            cosmo=self.astropy_cosmology)

        lenses = []
        for m, (x, y) in zip(masses, positions):
            # Convert mass and concentration to lensing parameters
            r_scale, r_scale_bending = lens_cosmo.nfw_physical2angle(
                M=m, c=concentration)

            subhalo = ('TNFW', {
                # Scale radius
                'Rs': r_scale,
                # Observed bending angle at scale radius
                'alpha_Rs': r_scale_bending,
                # Truncation radius
                'r_trunc': truncation * r_scale,
                'center_x': x, 'center_y': y})
            lenses.append(subhalo)
        return lenses

    def lensed_image(self, lenses=None, catalog_i=None, phi=None,
                     noise_seed=42, mask_radius=None):
        """Return numpy array describing lensed image

        Args:
            lenses: list of (lens model name, lens kwargs)
            catalog_i: image index from COSMOS catalog,
                If not provided, choose a random index.
            phi: rotation to apply to the source galaxy.
                If not provided, choose a random angle or 0,
                depending on manada's random_rotation option.
            noise_seed: (temporary) seed to use during noise generation.
                Set to 'random' to generate random noise (does not set seed),
                set to None to disable noise altogether.
            mask_radius: arcseconds in the center to zero out.
                If omitted, follow manada config.
        """
        if lenses is None:
            # Do not lens
            lenses = [('SIS', dict(theta_E=0.))]
        catalog_i, phi = \
            self.catalog.fill_catalog_i_phi_defaults(catalog_i, phi)

        lens_model_names, lens_kwargs, lens_zs = self._from_lens_list(lenses)
        if lens_zs is None:
            lens_model = ls.LensModel_(
                lens_model_names,
                z_lens=self.z_lens)
        else:
            lens_model = ls.LensModel_(
                lens_model_names,
                z_source=self.z_source,
                lens_redshift_list=lens_zs,
                cosmo=self.astropy_cosmology,
			    multi_plane=True)

        source_model_class, kwargs_source = self.catalog.draw_source(
            catalog_i=catalog_i,
            phi=phi,
            z_new=self.z_source)

        img = ls.ImageModel(
                # These arguments are called '...class', but expect instances!
                data_class=ls.ImageData(**ls.data_configure_simple(
                    numPix=self.n_pixels,
                    deltaPix=self.pixel_width)),
                psf_class=ls.Data.psf.PSF(
                    psf_type='GAUSSIAN',
                    fwhm=self.config_dict['psf']['parameters']['fwhm']),
                lens_model_class=lens_model,
                source_model_class=ls.LightModel_(source_model_class),
                kwargs_numerics=self.manada_config.kwargs_numerics,
            ).image(
                kwargs_lens=lens_kwargs,
                kwargs_source=kwargs_source)

        if noise_seed == 'random':
            img += self.single_band.noise_for_model(img)
        elif noise_seed is not None:
            with dds.temp_numpy_seed(noise_seed):
                img += self.single_band.noise_for_model(img)

        if mask_radius is None:
            mask_radius = getattr(self.manada_config, 'mask_radius', None)
        if mask_radius:
            x_grid, y_grid = np.meshgrid(*dds.image_grid(
                    img.shape,
                    pixel_width=self.pixel_width,
                    edges=False),
                indexing='ij')
            img[(x_grid**2 + y_grid**2) <= mask_radius**2] = 0

        return img

    def show_image(self, img, **kwargs):
        return dds.show_image(img, pixel_width=self.pixel_width, **kwargs)
