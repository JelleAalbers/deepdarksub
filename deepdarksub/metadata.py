from pathlib import Path

import numpy as np
import pandas as pd

import deepdarksub as dds
import manada
export, __all__ = dds.exporter()


def _load_csv(fn, filename_prefix=''):
    meta = pd.read_csv(fn)
    if 'index' not in meta:
        # Directly from manada
        meta['index'] = np.arange(len(meta))
        meta['filename'] = [filename_prefix + f'{x:07d}.npy'
                            for x in meta['index'].values]
        meta.set_index('index')
    else:
        print("Metadata is not directly from manada "
              "-- loading it as-is")
    return meta


@export
def load_bad_galaxies(filename=Path(__file__).parent / 'bad_galaxies.txt'):
    bad_indices = []
    # Load bad galaxies txt file
    with open(filename) as f:
        for l in f.readlines():
            l = l.strip().split('#')[0]
            if not l:
                continue
            try:
                l = int(l)
            except:
                print(f'Strange line, ignored: l')
            else:
                bad_indices.append(l)
    return np.sort(np.array(bad_indices, np.int))

@export
def load_metadata(
        data_dir,
        bad_galaxies=load_bad_galaxies(),
        val_galaxies=(6233, 10646, 12377, 17214, 25547, 39720, 43660, 51097),
        val_split='by_galaxy',
        filename_prefix='image_',
        remove_bad=True):
    """Load manada's metadata from data_dir

    :param data_dir: string or Path to manada dataset folder
    :param bad_galaxies: COSMOS catalog indices of galaxies to remove
    :param val_galaxies: COSMOS catalog indices of galaxies to use
        in the validation set
    :return: (
        metadata DataFrame,
        dict with sets of galaxy indices)
    """
    meta_fn = Path(data_dir) / 'metadata.csv'
    if meta_fn.exists():    
        meta = _load_csv(meta_fn, filename_prefix)
    else:
        # Multi-directory dataset
        metas = []
        for subdir in sorted(Path(data_dir).glob('*')):
            m = _load_csv(subdir / 'metadata.csv',
                          subdir.stem + '/' + filename_prefix)
            m['index_in_dir'] = m['index']
            metas.append(m)
        meta = pd.concat(metas, ignore_index=True)
        # Ensure we have a simple new index
        meta['index'] = np.arange(len(meta))
        meta.set_index('index')

    # Add extra columns from the source metadata
    try:
        lm = dds.LensMaker()   # quick access to cosmo and catalog
    except FileNotFoundError as e:
        print(f"Could not load lensmaker, COSMOS dataset missing? "
              f"Metadata will be incomplete. Original exception: {e}")
    else:
        cosmo = lm.catalog.cosmo
        cat_meta = lm.catalog.catalog[
            meta['source_parameters_catalog_i'].values.astype(np.int)]
        meta['source_z_orig'] = cat_meta['zphot']
        meta['source_z_scaling'] = (
                cosmo.angularDiameterDistance(meta['source_z_orig'])
                / cosmo.angularDiameterDistance(meta['source_parameters_z_source']))
        meta['source_scaled_flux_radius'] = (
                cat_meta['flux_radius']
                * meta['source_z_scaling']
                * manada.Sources.cosmos.HUBBLE_ACS_PIXEL_WIDTH)

        # Add Sersic fit info
        _fit_results = lm.catalog.catalog['sersicfit'].astype(np.float)
        sersic_params = 'intensity r_half n q boxiness x0 y0 phi'.split()
        sersic_info = {
            p: _fit_results[:, i]
            for i, p in enumerate(sersic_params)}
        for p in sersic_params:
            meta['source_parameters_sersicfit_' + p] = sersic_info[p][
                meta['source_parameters_catalog_i'].values.astype(np.int)]

    # Galaxy indices
    gis = dict()
    gis['all'] = np.unique(meta['source_parameters_catalog_i'].values.astype(np.int))
    gis['bad'] = np.array(bad_galaxies)
    gis['good'] = np.setdiff1d(gis['all'], gis['bad'])

    # Remove images with bad source galaxies
    meta['is_bad'] = np.in1d(meta['source_parameters_catalog_i'], gis['bad'])
    if remove_bad:
        meta = meta[~meta['is_bad']]
    print(f"{len(gis['all'])} distinct source galaxies used by manada. ")

    if val_split == 'by_galaxy':
        gis['val'] = np.array(val_galaxies)
        assert all([g in gis['all'] for g in gis['val']]), "Typo in val galaxy indices"
        gis['train'] = np.setdiff1d(gis['good'], gis['val'])

        # Label validation dataset
        meta['is_val'] = np.in1d(meta['source_parameters_catalog_i'], gis['val'])

        print(
          f"Throw away {len(gis['bad'])}, use {len(gis['val'])} for validation, "
          f"{len(gis['train'])} left for training.\n")

    elif val_split == 'random':
        with dds.temp_numpy_seed(42):
            meta['is_val'] = np.random.rand(len(meta)) < 0.1

    else:
        raise ValueError("Unrecognized val_split {val_split}")

    print(
      f"Total images: {len(meta)}; {len(meta) - meta['is_val'].sum()} for training and "
      f"{meta['is_val'].sum()} for validation.")

    # Default to uniform training weights
    if 'training_weight' not in meta.columns:
        meta['training_weight'] = 1.

    return meta, gis


@export
def filename_to_index(fn):
    return int(fn.stem.split('_')[-1])


@export
def meta_for_filename(meta, fn):
    return meta.loc[filename_to_index(fn)]
