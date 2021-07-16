import builtins
import contextlib
from datetime import datetime
import json
from functools import partial
from pathlib import Path
import tempfile

import fastai.vision.all as fv
import numpy as np
import torch

import deepdarksub as dds
export, __all__ = dds.exporter()
__all__.extend(['short_names'])


mdef = 'main_deflector_parameters_'
short_names = dict((
    (mdef + 'theta_E', 'theta_E'),
    ('subhalo_parameters_sigma_sub', 'sigma_sub'),
    ('los_parameters_delta_los', 'delta_los'),
    (mdef + 'center_x', 'center_x'),
    (mdef + 'center_y', 'center_y'),
    (mdef + 'gamma', 'gamma'),
    (mdef + 'gamma1', 'gamma1'),
    (mdef + 'gamma2', 'gamma2'),
    (mdef + 'e1', 'e1'),
    (mdef + 'e2', 'e2')))


@export
class Model:

    @classmethod
    def from_json(cls, filename, with_weights=True, **kwargs):
        """Build model using configuration from a json
        metadata file.

        Args:
            filename: full path to json
            with_weights: Try to load weights from .pth file in the same folder
            **kwargs: any options to override train_config
                from the json with.

        To setup pretrained models, you must also load the weights!
        """
        # The model json evolved from the training log
        r = load_training_log(filename)

        original_dataset = r['train_config']['dataset_name']
        if 'normalizer_means' in r:
            # Oops, should have put these in train_config
            r['train_config']['normalizer_means'] = r['normalizer_means']
            r['train_config']['normalizer_scales'] = r['normalizer_scales']
        elif 'normalizer_means' not in kwargs:
            print("Old json, normalizer settings omitted. Since model was "
                  f"trained for {original_dataset}, assuming its statistics "
                  "for normalization.")
            kwargs.update(normalizer_defaults(original_dataset))

        kwargs = {**r['train_config'], **kwargs}
        model = cls(**kwargs)

        model.training_log = r

        if with_weights:
            model_name = Path(filename).stem
            expected_loc = Path(model.learner.model_dir) / (model_name + '.pth')
            if expected_loc.exists():
                model.learner.load(model_name)
            else:
                raise ValueError(
                    f"Missing weights file at {expected_loc}; use "
                    "with_weights=False to leave weights to (random) "
                    "initial values")

        return model


    def __init__(self,
                 verbose=True,
                 base_dir='.',
                 test_only=False,
                 **kwargs):
        """Initialize substructure-predicting model

        Args:
            verbose: if True, print messages during initialization
            base_dir: Path to directory containing datasets (each in their
                own folder). Defaults to current directory.
            test_only: if True, initialize with a dummy dataset
                (2 blank images with meaningless metadata).
            **kwargs: Configuration options
        """
        self.print = print = builtins.print if verbose else lambda x: x

        self.train_config = tc = dict(
            val_galaxies = dds.metadata.val_galaxies,
            bad_galaxies = dds.load_bad_galaxies())
        tc.update(**kwargs)

        self.fit_parameters = tc['fit_parameters']
        self.n_params = len(self.fit_parameters)
        self.short_names = [dds.short_names[pname]
                            for pname in self.fit_parameters]

        if test_only:
            print(f"Setting up model with meaningless toy data. You won't be "
                  "able to train or use predict_all, but you can run predict "
                  "on new images.")
            self.data_dir = dds.make_dummy_dataset()
        else:
            self.data_dir = Path(base_dir) / tc['dataset_name']
            if self.data_dir.exists():
                print(f"Setting up model for dataset {tc['dataset_name']}")
            else:
                raise FileNotFoundError(
                    f"{self.data_dir} not found! Check base dir, or "
                    "pass test_only = True to setup model for evaluation only.")

        self.metadata, self.galaxy_indices = dds.load_metadata(
            self.data_dir,
            val_galaxies=tc['val_galaxies'],
            bad_galaxies=tc['bad_galaxies'],
            remove_bad=True,
            verbose=False if test_only else verbose)
        if 'normalizer_means' in tc:
            self.normalizer = dds.Normalizer(
                fit_parameters=self.fit_parameters,
                means=tc['normalizer_means'],
                scales=tc['normalizer_scales'])
        else:
            self.normalizer = dds.Normalizer(self.metadata, self.fit_parameters)

        self.has_cuda = torch.cuda.is_available()
        print(f"Cuda available: {self.has_cuda}")
        if self.has_cuda:
            print("CUDA device: "
                + torch.cuda.get_device_name(torch.cuda.current_device()))

        # Setting these up will take a while; looks like it's loading the entire
        # dataset in RAM? I'm probably butchering the fastai dataloader API...
        print("Setting up data block and data loaders, could take a while")
        self.data_block = dds.data_block(
            self.metadata,
            fit_parameters=tc['fit_parameters'],
            data_dir=self.data_dir,
            uncertainty=tc['uncertainty'],
            augment_rotation=tc['augment_rotation'])
        self.data_loaders = self.data_block.dataloaders(None,
                                                        bs=tc['batch_size'])
        print("Dataloaders initialized")

        self.metrics = dds.all_metrics(
            self.fit_parameters,
            self.normalizer,
            self.short_names,
            self.train_config['uncertainty'])

        arch = getattr(fv, tc['architecture'])
        if 'architecture_options' in tc:
            arch = partial(arch, **tc['architecture_options'])

        self.dropout_switch = dds.TestTimeDropout()

        if tc.get('truncate_final_to') is None:
            truncate_final_to = None
        else:
            # Truncate final parameter to physical value = 0;
            # find encoded value of zero
            final_p = list(self.fit_parameters)[-1]
            truncate_final_to = self.normalizer.norm(0, param_name=final_p)
            print(f"Truncating {final_p} to 0, encoded as {truncate_final_to}")

        self.learner = fv.cnn_learner(
            dls=self.data_loaders,
            arch=arch,
            n_in=1,
            n_out=dds.n_out(self.n_params, tc['uncertainty']),
            loss_func=dds.loss_for(
                self.fit_parameters,
                tc['uncertainty'],
                truncate_final_to=truncate_final_to,
                parameter_weights=tc.get('parameter_weights')),
            metrics=self.metrics,
            pretrained=False,
            cbs=[self.dropout_switch],
            bn_final=tc['bn_final'])

    def predict(self,
                image,
                as_dict=True,
                short_names=True,
                with_dropout=False,
                **kwargs):
        """Return (prediction, uncertainty) for a single image

        Args:
            image: str/Path to npy image, or numpy array
            as_dict: If True, ... returns dicts of floats, else array
            short_names: If True, dicts will use short-form parameter names
            with_dropout: If True, activate dropout
                (will partially randomizes prediction)
        """
        if isinstance(image, np.ndarray):
            with tempfile.NamedTemporaryFile() as tempf:
                np.save(tempf, image)
                return self.predict(
                    image=tempf.name,
                    as_dict=as_dict,
                    short_names=short_names,
                    with_dropout=with_dropout,
                    **kwargs)

        pred, unc = self.predict_many(
            [image],
            progress=False,
            as_dict=as_dict,
            short_names=short_names,
            with_dropout=with_dropout,
            **kwargs)
        if as_dict:
            # No need to return 1-element arrays, just extract the float
            pred = {k: v[0] for k, v in pred.items()}
            if isinstance(unc, dict):
                unc = {k: v[0] for k, v in unc.items()}
        else:
            pred = pred[0]
        if isinstance(unc, np.ndarray) and len(unc.shape) == 3:
            # Remove the batch dimension, we're just predicting one image
            unc = unc[0]
        return pred, unc

    def predict_many(self,
                     filenames,
                     progress=True,
                     as_dict=True,
                     short_names=True,
                     with_dropout=False,
                     **kwargs):
        """Return (predictions, uncertainties) for images in filenames list

        Args:
            filenames: sequence of str/Path to .npy images
            progress: if True (default), show a progress bar
            as_dict: If True, ... returns dict of arrays (one per param),
                otherwise 2d arrays (param order matches self.fit_parameters).
            with_dropout: If True, activate dropout
                (will partially randomizes prediction)
        """
        if isinstance(filenames, (str, Path)):
            raise ValueError(
                "Expected a sequence of filenames; have you seen .predict?")
        filenames = [Path(fn) for fn in filenames]
        dl = self.learner.dls.test_dl(filenames, **kwargs)
        nullc = contextlib.nullcontext
        with nullc() if progress else self.learner.no_bar():
            with self.dropout_switch.active(with_dropout):
                preds = self.learner.get_preds(dl=dl)[0]
        y_pred, y_unc = self.normalizer.decode(
            preds,
            as_dict=as_dict,
            uncertainty=self.train_config['uncertainty'],)
        if short_names and as_dict:
            y_pred = self._shorten_dict(y_pred)
            if isinstance(y_unc, dict):
                self._shorten_dict(y_unc)
        return y_pred, y_unc

    def predict_all(self,
                    dataset='val',
                    as_dict=True,
                    short_names=True):
        """Return (pred=..., unc=..., true=...) for validation or training data.

        Args:
            dataset: 'train' gets training data results, 'val' (default)
                validation data.
            as_dict: If True, ... in return value is a dict of arrays,
                (one per param), otherwise a 2d array (param order matches
                self.fit_parameters).
            short_names: If True, dicts will use short-form parameter names
        """
        preds, targets = self.learner.get_preds(
            ds_idx=0 if dataset == 'train' else 1,
            reorder=False)
        y_pred, y_unc = self.normalizer.decode(
            preds,
            uncertainty=self.train_config['uncertainty'],
            as_dict=as_dict)
        y_true, _ = self.normalizer.decode(
            targets[:,:self.n_params],
            as_dict=as_dict)
        return {
            label: self._shorten_dict(x) if as_dict and short_names and isinstance(x, dict)
                   else x
            for label, x in [('pred', y_pred),
                             ('unc', y_unc),
                             ('true', y_true)]}

    @staticmethod
    def _shorten_dict(x):
        return {dds.short_names[pname]: val
                for pname, val in x.items()}

    def train(self, model_dir='models', lr_find=True):
        """Train the model according to the configuration, then
        save model (.pth) and training log (.json) in results_dir"""
        # Get a unique name from the current time and configuration,
        # for naming the model and training logs,
        result_name = (
            datetime.now().strftime('%y%m%d_%H%M')
            + '_' + self.train_config['dataset_name']
            + '_' + dds.deterministic_hash(self.train_config))
        print(f"Starting training; results will be saved as {result_name}")

        if lr_find:
            print("Running lr_find")
            self.learner.lr_find(show_plot=True)

            import matplotlib.pyplot as plt
            Path('./plots').mkdir(exist_ok=True)
            plt.axvline(self.train_config['base_lr'], color='red', linewidth=1)
            plot_fname = 'plots/' + result_name + '_lr_find.png'
            plt.savefig(plot_fname, dpi=200, bbox_inches='tight')
            print(f"Saved lr_find plot to {plot_fname}")

        self.learner.fit_one_cycle(
            n_epoch=self.train_config['n_epochs'],
            lr_max=self.train_config['base_lr'],
            **self.train_config.get('lr_schedule', {}))
        print("Training finished")

        model_dir = Path(model_dir)
        model_dir.mkdir(exist_ok=True)
        self.learner.model_dir = str(model_dir)

        out = dict(zip(
            ['train_loss', 'val_loss', *[f.name for f in self.metrics]],
            np.stack(self.learner.recorder.values).T.tolist()))
        n_images = len(self.metadata)
        n_val = self.metadata['is_val'].sum()
        out.update(
            train_loss_hr = [x.numpy().item()
                            for x in self.learner.recorder.losses],
            # only last epoch duration is recorded... oh well
            epoch_duration = self.learner.recorder.log[-1],
            n_images = n_images,
            n_validation_images = n_val,
            n_training_images = n_images - n_val,
            train_config=self.train_config)
        # Store normalizer config, so we can use the model on other datasets
        out.update(dict(
            normalizer_means=self.normalizer.means,
            normalizer_scales=self.normalizer.scales))

        with open(model_dir / (result_name + '.json'), mode='w') as f:
            json.dump(out, f, cls=dds.NumpyJSONEncoder)

        self.learner.save(result_name)
        return result_name


@export
def load_training_log(fn):
    """Return info from the training log / model json filename fn"""
    with open(fn) as f:
        r = json.load(f)

    r['short_names'] = short_names = [
        dds.short_names[pname]
        for pname in r['train_config']['fit_parameters']]
    # Collect training data metrics of the same kind together
    ms = dict()
    for metric in 'rmse', 'unc', 'rho', 'sigma_sub_rho':
        if (r['short_names'][0] + '_' + metric) not in r:
            continue
        ms[metric] = {pname: np.array(r[pname + '_' + metric])
                      for pname in short_names}
    r['metrics'] = ms

    n_epochs = r['train_config']['n_epochs']
    # Old logs did not store n_training images;
    # 0.956 is the training fraction for dl_ss_npy and more_500k.
    n_training = r.get('n_training_images', r['n_images'] * 0.956)
    r['training_images_seen'] = np.arange(n_epochs) * n_training / 1e6

    return r


@export
def normalizer_defaults(dataset_name):
    """Get normalizer defaults for dataset name

    Legacy code, will eventually be removed; new models have this info
        in their jsons!
    """
    if dataset_name == 'dl_ss_npy':
        return {
            'normalizer_means': {
                'main_deflector_parameters_theta_E': 1.105562778713642,
                'subhalo_parameters_sigma_sub': 0.11330676025232005,
                'los_parameters_delta_los': 1.2786016271288887,
                'main_deflector_parameters_center_x': 0,
                'main_deflector_parameters_center_y': 0,
                'main_deflector_parameters_gamma': 2.012546764682251,
                'main_deflector_parameters_gamma1': 0,
                'main_deflector_parameters_gamma2': 0,
                'main_deflector_parameters_e1': 0,
                'main_deflector_parameters_e2': 0},
            'normalizer_scales': {
                'main_deflector_parameters_theta_E': 0.11083102731205988,
                'subhalo_parameters_sigma_sub': 0.060380880509218304,
                'los_parameters_delta_los': 1.015695530279778,
                'main_deflector_parameters_center_x': 1,
                'main_deflector_parameters_center_y': 1,
                'main_deflector_parameters_gamma': 0.10077831063987681,
                'main_deflector_parameters_gamma1': 1,
                'main_deflector_parameters_gamma2': 1,
                'main_deflector_parameters_e1': 1,
                'main_deflector_parameters_e2': 1}}

    elif dataset_name == 'more_500k':
        return {
            'normalizer_means': {
                'main_deflector_parameters_theta_E': 1.1056347236788366,
                'subhalo_parameters_sigma_sub': 0.09976644381301354,
                'los_parameters_delta_los': 1.000093967445753,
                'main_deflector_parameters_center_x': 0,
                'main_deflector_parameters_center_y': 0,
                'main_deflector_parameters_gamma': 2.012380520111582,
                'main_deflector_parameters_gamma1': 0,
                'main_deflector_parameters_gamma2': 0,
                'main_deflector_parameters_e1': 0,
                'main_deflector_parameters_e2': 0},
            'normalizer_scales': {
                'main_deflector_parameters_theta_E': 0.11093302082062602,
                'subhalo_parameters_sigma_sub': 0.05770832453252406,
                'los_parameters_delta_los': 0.5770424745804907,
                'main_deflector_parameters_center_x': 1,
                'main_deflector_parameters_center_y': 1,
                'main_deflector_parameters_gamma': 0.1006303388806144,
                'main_deflector_parameters_gamma1': 1,
                'main_deflector_parameters_gamma2': 1,
                'main_deflector_parameters_e1': 1,
                'main_deflector_parameters_e2': 1}}
