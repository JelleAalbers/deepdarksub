import builtins
import contextlib
from datetime import datetime
import json
from pathlib import Path
import tempfile

import fastai.vision.all as fv
import numpy as np
import torch

import deepdarksub as dds
export, __all__ = dds.exporter()
__all__.extend(['shorten_param_name'])


mdef = 'main_deflector_parameters_'
shorten_param_name = dict((
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
    def from_json(cls, filename, **kwargs):
        """Build model using configuration from a json
        metadata file.
        
        Args:
            filename: full path to json
            **kwargs: any options to override train_config
                from the json with.
                
        To setup pretrained models, you must also load the weights!
        """
        with open(filename) as f:
            r = json.load(f)
        original_dataset = r['train_config']['dataset_name']
        kwargs = {**r['train_config'], **kwargs}
        if 'normalizer_means' not in kwargs:
            print("Old json, normalizer settings omitted. Since model was "
                  f"trained for {original_dataset}, assuming its statistics "
                  "for normalization.")
            kwargs.update(normalizer_defaults(original_dataset))
        return cls(**kwargs)

    def __init__(self,
                 verbose=True,
                 base_dir='.',
                 toy_data=False,
                 **kwargs):
        """Initialize substructure-predicting model

        Args:
            verbose: if True, print messages during initialization
            base_dir: Path to directory containing datasets (each in their
                own folder). Defaults to current directory.
            toy_data: if True, initialize with a dummy dataset
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
        self.short_names = [shorten_param_name[pname]
                            for pname in self.fit_parameters]


        if not toy_data:
            self.data_dir = Path(base_dir) / tc['dataset_name']
            if self.data_dir.exists():
                print(f"Setting up model for dataset {tc['dataset_name']}")
            else:
                print(f"{self.data_dir} not found, using toy dataset instead")
                toy_data = True
        if toy_data:
            print(f"Setting up model with meaningless toy data")
            self.data_dir = dds.make_dummy_dataset()

        self.metadata, self.galaxy_indices = dds.load_metadata(
            self.data_dir,
            val_galaxies=tc['val_galaxies'],
            bad_galaxies=tc['bad_galaxies'],
            remove_bad=True,
            verbose=verbose)
        if 'normalizer_means' in tc:
            self.normalizer = dds.Normalizer(
                fit_parameters=self.fit_parameters,
                means=tc['normalizer_means'],
                scales=tc['normalizer_scales'])
        else:
            self.normalizer = dds.Normalizer(self.metadata, self.fit_parameters)

        print(f"Cuda available: {torch.cuda.is_available()}")
        print("CUDA device: "
            + torch.cuda.get_device_name(torch.cuda.current_device()))

        # Setting these up will take a while; looks like it's loading the entire
        # dataset in RAM? I'm probably butchering the fastai dataloader API...
        self.data_block = dds.data_block(
            self.metadata,
            fit_parameters=tc['fit_parameters'],
            data_dir=self.data_dir,
            uncertainty=tc['uncertainty'],
            augment_rotation=tc['augment_rotation'])
        self.data_loaders = self.data_block.dataloaders(None,
                                                        bs=tc['batch_size'])
        print("Dataloaders initialized")

        if tc['uncertainty'] == 'diagonal':
            self.metrics = dds.all_metrics(
                self.fit_parameters, self.normalizer, self.short_names)
        else:
            print("Training metrics require diagonal uncertainty (for now)")

        self.learner = fv.cnn_learner(
            dls=self.data_loaders,
            arch=getattr(fv, tc['architecture']),
            n_in=1,
            n_out=dds.n_out(self.n_params, tc['uncertainty']),
            loss_func=dds.loss_for(
                self.fit_parameters,
                tc['uncertainty'],
                parameter_weights=tc.get('parameter_weights')),
            metrics=self.metrics,
            pretrained=False,
            bn_final=tc['bn_final'])

    def predict_many(self,
                     filenames,
                     progress=True,
                     as_dict=True,
                     short_names=True,
                     **kwargs):
        """Return (predictions, uncertainties) for images in filenames list
        """
        if isinstance(filenames, (str, Path)):
            filenames = [filenames]
        dl = self.learner.dls.test_dl(filenames, **kwargs)
        with contextlib.nullcontext() if progress else self.learner.no_bar():
            preds = self.learner.get_preds(dl=dl)[0]
        y_pred, y_unc = self.normalizer.decode(
            preds,
            as_dict=as_dict,
            uncertainty=self.train_config['uncertainty'])
        if short_names:
            return self._shorten_dict(y_pred), self._shorten_dict(y_unc)
        return y_pred, y_unc

    def predict_all(self,
                    dataset='val',
                    as_dict=True,
                    short_names=True):
        """Return (pred=..., unc=..., true=...) for validation or training data

        Args:
            dataset: 'train' gets training data predictions, 'val' validation data
            as_dict: If True, ... is a dict of arrays (one per param),
                otherwise a 2d array (parameter order matches self.fit_parameters)
            short_names: If True, dicts will use short-form parameter names
        """
        preds, targets = self.learner.get_preds(
            ds_idx=0 if dataset == 'train' else 1,
            reorder=False)
        y_pred, y_unc = self.normalizer.decode(
            preds,
            uncertainty=self.train_config['uncertainty'],
            as_dict=as_dict)
        y_true, _ = self.normalizer.decode(targets[:,:self.n_params],
                                        as_dict=as_dict)
        return {
            label: self._shorten_dict(x) if as_dict and short_names else x
            for label, x in (
                ('pred', y_pred), ('unc', y_unc), ('true', y_true))}

    @staticmethod
    def _shorten_dict(x):
        return {shorten_param_name[pname]: val
                for pname, val in x.items()}

    def train(self, model_dir='models'):
        """Train the model according to the configuration, then
        save model (.pth) and training log (.json) in results_dir"""
        # Get a unique name from the current time and configuration,
        # for naming the model and training logs,
        result_name = (
            datetime.now().strftime('%y%m%d_%H%M')
            + '_' + self.train_config['dataset_name']
            + '_' + dds.deterministic_hash(self.train_config))
        print(f"Starting training; results will be saved as {result_name}")

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
        out.update(
            train_loss_hr = [x.numpy().item()
                            for x in self.learner.recorder.losses],
            # only last epoch duration is recorded... oh well
            epoch_duration = self.learner.recorder.log[-1],
            n_images = len(self.meta),
            train_config=self.train_config)
        # Store normalizer config, so we can use the model on other datasets
        out.update(dict(
            normalizer_means=self.normalizer.means,
            normalizer_scales=self.normalizer.scales))

        with open(model_dir / (result_name + '.json'), mode='w') as f:
            json.dump(out, f)

        self.learner.save(result_name)



@export
def normalizer_defaults(dataset_name):
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
