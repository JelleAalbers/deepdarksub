import builtins
from pathlib import Path

import numpy as np
import pandas as pd

import deepdarksub as dds
import manada
export, __all__ = dds.exporter()


def _load_csv(fn, filename_prefix=''):
    meta = pd.read_csv(fn)
    if 'filename' not in meta:
        # Directly from manada: add filename
        meta['image_number'] = np.arange(len(meta))
        meta['filename'] = [filename_prefix + f'{x:07d}.npy'
                            for x in meta['image_number'].values]
    meta = meta.sort_values(by='filename')
    meta = meta.set_index('filename')
    meta['filename'] = meta.index
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


val_galaxies = (
         548,   808,   882,  1601,  2093,  2383,  2668,  2771,  3579,
        3637,  4483,  4569,  5215,  5753,  5872,  6389,  6580,  6985,
        7212,  7731,  7775,  7987,  8016,  8525,  8550,  9295, 12896,
       13806, 14190, 14193, 14355, 14868, 16373, 16887, 16957, 17258,
       17445, 17675, 17793, 17869, 18035, 21593, 22088, 22192,
       22511, 22545, 22780, 24221, 24784, 25041, 25335, 25936, 26041,
       26231, 26259, 26502, 27342, 27887, 28995, 29099, 29134, 29911,
       32081, 32141, 32491, 32883, 35415, 36475, 36540, 39296, 39913,
       40085, 40316, 40501, 41570, 41784, 42484, 42606, 42635, 44780,
       44826, 47839, 47903, 48239, 48869, 49719, 49949, 50020, 50424,
       50446, 50578, 50761, 51021, 51047, 51073, 52424, 53227, 53821,
       54056
)


@export
def load_metadata(
        data_dir,
        bad_galaxies=load_bad_galaxies(),
        val_galaxies=val_galaxies,
        val_split='by_galaxy',
        filename_prefix='image_',
        verbose=True,
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
    print = builtins.print if verbose else lambda x: x

    meta_fn = Path(data_dir) / 'metadata.csv'
    if meta_fn.exists():
        meta = _load_csv(meta_fn, filename_prefix)
    else:
        # Multi-directory dataset
        metas = []
        for subdir in sorted(Path(data_dir).glob('*')):
            m = _load_csv(subdir / 'metadata.csv',
                          subdir.stem + '/' + filename_prefix)
            metas.append(m)
        # Concatenation seems to remove the filename column
        # (maybe because it is also the index?)
        # (ignore_index is worse, that would remove filename completely.)
        meta = pd.concat(metas)
#         meta = meta.reset_index()  # Makes filename a regular column again?!
#         meta = meta.sort_values(by='filename')
#         meta = meta.set_index('filename')
#         meta['filename'] = meta.index

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
        unknown_val = sorted(list(set(gis['val']) - set(gis['all'])))
        if unknown_val:
            print(f"Validation galaxies {unknown_val} do not occur in metadata")
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
        raise ValueError(f"Unrecognized val_split {val_split}")

    print(
      f"Total images: {len(meta)}; {len(meta) - meta['is_val'].sum()} for training and "
      f"{meta['is_val'].sum()} for validation.")

    # Default to uniform training weights
    if 'training_weight' not in meta.columns:
        meta['training_weight'] = 1.

    return meta, gis


@export
def meta_for_filename(meta, fn):
    # TODO: simplify get_items so we don't
    # need to do so much here

    # Do we have a subfolder-based dataset?
    components = str(fn).split('/')
    subfolder, image_fn = components[-2:]
    try:
        int(subfolder)
    except:
        is_subfolder = False
    else:
        is_subfolder = True

    # Build the filename key
    if is_subfolder:
        fn = subfolder + '/' + image_fn
    else:
        fn = image_fn

    # Assume filename is the index
    return meta.loc[fn]
