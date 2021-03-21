from pathlib import Path

import numpy as np
import pandas as pd

import deepdarksub as dds
import manada
export, __all__ = dds.exporter()


@export
def load_metadata(
        data_dir,
        bad_galaxies=(36412, 53912, 53954),
        val_galaxies=(6233, 10646, 12377, 17214, 25547, 39720, 43660, 51097),
        remove_bad=True):
    """Load manada's metadata from data_dir

    :param data_dir: string or Path to manada dataset folder
    :param bad_galaxies: COSMOS catalog indices of galaxies to remove
    :param val_galaxies: COSMOS catalog indices of galaxies to use
    :return: (
        metadata DataFrame,
        dict with sets of galaxy indices)
    """
    meta = pd.read_csv(Path(data_dir) / 'metadata.csv')
    if 'index' not in meta:
        # Directly of manada, has not passed through merger script
        meta['index'] = np.arange(len(meta))
        meta['filename'] = [f'image_{x:07d}.npy' for x in meta['index'].values]
    meta.set_index('index')

    # Add extra columns from the source metadata
    lm = dds.LensMaker()   # quick access to cosmo and catalog
    cosmo = lm.catalog.cosmo
    cat_meta = lm.catalog.catalog[
        meta['source_parameters_catalog_i'].values.astype(np.int)]
    meta['source_z_orig'] = cat_meta['zphot']
    meta['source_z_scaling'] = (
            cosmo.angularDiameterDistance(meta['source_z_orig'])
            / cosmo.angularDiameterDistance(lm.z_source))
    meta['source_scaled_flux_radius'] = (
            cat_meta['flux_radius']
            * meta['source_z_scaling']
            * manada.Sources.cosmos.HUBBLE_ACS_PIXEL_WIDTH)

    # Galaxy indices
    gis = dict()
    gis['all'] = np.unique(meta['source_parameters_catalog_i'].values.astype(np.int))

    gis['bad'] = np.array(bad_galaxies)
    assert all([g in gis['all'] for g in gis['bad']]), "Typo in bad galaxy indices"
    gis['good'] = np.setdiff1d(gis['all'], gis['bad'])
    gis['val'] = np.array(val_galaxies)
    assert all([g in gis['all'] for g in gis['val']]), "Typo in val galaxy indices"
    gis['train'] = np.setdiff1d(gis['good'], gis['val'])

    # Remove images with bad source galaxies
    meta['is_bad'] = np.in1d(meta['source_parameters_catalog_i'], gis['bad'])
    if remove_bad:
        meta = meta[~meta['is_bad']]

    # Label validation dataset
    meta['is_val'] = np.in1d(meta['source_parameters_catalog_i'], gis['val'])

    print(f"{len(gis['all'])} distinct source galaxies used by manada. "
          f"Throw away {len(gis['bad'])}, use {len(gis['val'])} for validation, "
          f"{len(gis['train'])} left for training.\n"
          f"Total images: {len(meta)}; {len(meta) - meta['is_val'].sum()} for training and "
          f"{meta['is_val'].sum()} for validation.")

    return meta, gis


@export
def filename_to_index(fn):
    return int(fn.stem.split('_')[1])


@export
def meta_for_filename(meta, fn):
    return meta.loc[filename_to_index(fn)]
