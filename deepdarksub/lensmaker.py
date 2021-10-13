import contextlib

import lenstronomy as ls
import numpy as np
import pandas as pd

from manada import generate as manada_generate
from manada.Sampling.sampler import Sampler
from manada.Utils.cosmology_utils import get_cosmology

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
        models = ('PEMD', 'SHEAR')   # TODO: nicer not to hardcode...
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

    def manada_substructure(self,
            sigma_sub=0.1,
            delta_los=1.,
            los_mode='subtract_average',
            **main_deflector_parameters):
        """Return (list of subhalo lenses, dataframe with metadata)

        See manada_los and manada_subhalos for arguments
        """
        los_lenses, los_meta = self.manada_los(
            delta_los=delta_los, mode=los_mode, **main_deflector_parameters)
        subhalo_lenses, subhalo_meta = self.manada_subhalos(
            sigma_sub=sigma_sub, **main_deflector_parameters)

        # Collect subhalo metadata
        df_sub = pd.DataFrame([l[1] for l in subhalo_lenses])
        if len(df_sub):
            df_sub['m'] = subhalo_meta['mass'][0][2]
            df_sub['c'] = subhalo_meta['concentration'][0][2]
            df_sub['z'] = self.z_lens
        else:
            df_sub['m'] = df_sub['c'] = df_sub['z'] = []

        # Collect LOS metadata
        _from_los = []
        for args, _, result in los_meta['concentration']:
            masses = args[1]
            if not isinstance(masses, np.ndarray):
                # Some c_average call
                continue
            for m, c in zip(masses, result):
                _from_los.append(dict(m=m, c=c))
                _from_los[-1].update(los_lenses[len(_from_los) - 1][1])

        return (
            (subhalo_lenses + los_lenses),
            pd.concat([df_sub, pd.DataFrame(_from_los)]).reset_index(drop=True))

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

        # Steal mass/concentration info from manada...
        spy_reports = dict()
        dds.spy_on_method(los_maker, 'draw_nfw_masses',
                          spy_reports, 'mass')
        dds.spy_on_method(los_maker, 'mass_concentration',
                          spy_reports, 'concentration')

        result = []
        if mode in ['halos', 'subtract_average']:
            result.append(los_maker.draw_los())
        if mode in ['average_only', 'subtract_average']:
            # TODO: ask Sebastian about * 2 here
            result.append(los_maker.calculate_average_alpha(self.n_pixels * 2))
        return sum([self._to_lens_list(*x) for x in result], []), spy_reports

    def manada_subhalos(self,
                        sigma_sub=0.1,
                        **main_deflector_parameters):
        """Return list of subhalo lenses (model, kwargs pairs), metadata dict"""
        c = self.config_dict
        subhalo_maker = c['subhalo']['class'](
			subhalo_parameters={**c['subhalo']['parameters'],
                                'sigma_sub': sigma_sub},
            **self._common_substructure_kwargs(main_deflector_parameters))

        # Steal mass/concentration info from manada...
        spy_reports = dict()
        dds.spy_on_method(subhalo_maker, 'draw_nfw_masses',
                          spy_reports, 'mass')
        dds.spy_on_method(subhalo_maker, 'mass_concentration',
                          spy_reports, 'concentration')

        sub_model_list, sub_kwargs_list, _ = subhalo_maker.draw_subhalos()
        return self._to_lens_list(sub_model_list, sub_kwargs_list), spy_reports

    def simple_subhalos(self, masses, positions, concentration=15, truncation=5):
        """Return (lenses, meta) for subhalos

        Args:
         - masses: sequence of subhalo masses
         - positions: sequence of (x,y) subhalo positions
         - concentration: concentration to use
         - truncation: TNFW r_trunc/r_scale

        Concentration=15, truncation=5 defaults come from
            https://arxiv.org/pdf/2009.06639.pdf
        """
        # LensCosmo handles cosmology-dependent parameter conversions
        lens_cosmo = ls.LensCosmo(
            z_lens=self.z_lens,
            z_source=self.z_source,
            cosmo=self.astropy_cosmology)

        lenses = []
        meta = []
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
            meta.append(subhalo[1])
            lenses.append(subhalo)

        df = pd.DataFrame(meta)
        df['z'] = self.z_lens
        df['m'] = m
        df['c'] = concentration
        return lenses, df

    def lens_model_kwargs(self, lenses):
        """Return lenstronomy (LensModel, kwargs) corresponding to Lenses"""
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
        return lens_model, lens_kwargs

    def lensed_image(self, lenses=None, catalog_i=None, phi=None,
                     noise_seed='random', mask_radius=None):
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

        lens_model, lens_kwargs = self.lens_model_kwargs(lenses)

        source_model_class, kwargs_source = self.catalog.draw_source(
            catalog_i=catalog_i,
            phi=phi)

        # Monkeypatch manada's draw_image with our own routine,
        # so we don't have to duplicate the stuff in draw_drizzled_image.
        def draw_image(
                # Optional snarky comment about keyword arguments here
                sample,los_class,subhalo_class,main_deflector_class,
                source_class,numpix,multi_plane,kwargs_numerics,mag_cut,add_noise,
                apply_psf=True):
            img = ls.ImageModel(
                    data_class=ls.DataAPI(numpix=numpix, **sample['detector_parameters']).data_class,
                    psf_class=ls.Data.psf.PSF(**(sample['psf_parameters'] if apply_psf else dict(psf_type='NONE'))),
                    lens_model_class=lens_model,
                    source_model_class=ls.LightModel_(source_model_class),
                    kwargs_numerics=kwargs_numerics,
                ).image(
                    kwargs_lens=lens_kwargs,
                    kwargs_source=kwargs_source)
            if add_noise:
                img += self.single_band.noise_for_model(img)
            return img, None

        if noise_seed == 'random':
            noise_context = contextlib.nullcontext
        else:
            noise_context = dds.temp_numpy_seed(noise_seed)

        vanilla_draw_image = manada_generate.draw_image
        with noise_context:
            try:
                manada_generate.draw_image = draw_image
                img, _ = manada_generate.draw_drizzled_image(
                    sample=Sampler(self.config_dict).sample(),
                    los_class=None,
                    subhalo_class=None,
                    main_deflector_class=None,
                    source_class=None,
                    numpix=self.n_pixels,
                    multi_plane=True,
                    kwargs_numerics=self.manada_config.kwargs_numerics,
                    mag_cut=None,
                    add_noise=noise_seed is not None)
            finally:
                manada_generate.draw_image = vanilla_draw_image

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
