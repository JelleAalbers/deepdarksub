#!/usr/bin/env python
import argparse

parser = argparse.ArgumentParser(description='Train a substructure predicting neural net')
parser.add_argument(
    '--finetune', metavar='netname', default=None,
    help="Train a head onto another net's frozen body")
parser.add_argument(
    '--epochs', type=int, default=1,
    help="Number of training epochs (default 1)")
parser.add_argument(
    '--batch_size', type=int, default=256,
    help="Batch size to use")
parser.add_argument(
    '--widen', type=float, default=1.,
    help="How much to widen the network")
parser.add_argument(
    '--uncertainty', default='diagonal',
    help='Kind of uncertainty the net should output: diagonal or correlated')
parser.add_argument(
    '--architecture', default='xresnet34',
    help='Network architecture to use')
parser.add_argument(
    '--dataset', default='acs_100k_dds',
    help='Dataset to use, must be a .zip or .tar in SCRATCH, or dir in LSCRATCH')
parser.add_argument(
    '--log', nargs='+',
    help='Short names of fit parameters for which NN should '
         'predict log')
parser.add_argument(
    '--lr', default=0.1, type=float,
    help='Base learning rate to use')
parser.add_argument(
    '--dropout_p', default=0.5, type=float,
    help='Dropout to use in the final layer (fastAI default is 0.5)')
parser.add_argument(
    '--truncate_final', action='store_true',
    help="Truncate the final parameter's posterior to be >= 0")
parser.add_argument(
    '--pct_start', default=0.3, type=float,
    help='Fraction of training to use for warm-up (ascending LR)')
args = parser.parse_args()

# Imports take a while on SDF due to the network filesystem.
print("Imports started")
import os
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import deepdarksub as dds
print("Imports done")

SCRATCH = os.getenv('SCRATCH') or Path('.')    # Where training data zips are
LSCRATCH = os.getenv('LSCRATCH') or Path('.')  # Where to unzip (will make subfolder)

Path('./plots').mkdir(exist_ok=True)


fit_parameters = (
    'main_deflector_parameters_center_x',
    'main_deflector_parameters_center_y',
    'main_deflector_parameters_gamma',
    'main_deflector_parameters_gamma1',
    'main_deflector_parameters_gamma2',
    'main_deflector_parameters_e1',
    'main_deflector_parameters_e2',
    'main_deflector_parameters_theta_E',
    #'los_parameters_delta_los',
    #'subhalo_parameters_shmf_plaw_index',
    'subhalo_parameters_sigma_sub')

params_as_inputs = (
    'main_deflector_parameters_z_lens',
    'source_parameters_z_source'
)
# Use short names when possible
fit_parameters = [
    dds.short_names.get(p, p)
    for p in fit_parameters]
if args.log:
    fit_parameters = [
        'log_' + p if p in args.log else p
        for p in fit_parameters]

for pname in fit_parameters:
    if 'sigma_sub' in pname:
        # Give sigma_sub 10x weight in the loss
        # (only if diagonal or no unc.)
        parameter_weights = dict(pnam =10)
        break
else:
    # No sigma_sub
    parameter_weights = dict()


train_config = dict(
    dataset_name = args.dataset,
    fit_parameters = fit_parameters,
    params_as_inputs = params_as_inputs,
    uncertainty = args.uncertainty,
    augment_rotation = 'free',
    batch_size = args.batch_size,
    truncate_final = args.truncate_final,

    parameter_weights = parameter_weights,

    lr_schedule = {
        'pct_start': args.pct_start
    },
    architecture_options = {
        'widen': args.widen
    },

    n_epochs = args.epochs,
    architecture = args.architecture,
    bn_final = True,
    dropout_p = args.dropout_p,
    base_lr = args.lr)

if train_config['uncertainty'] == 'correlated':
    # Haven't implemented weighting yet for correlated loss
    del train_config['parameter_weights']

data_dir = Path(LSCRATCH) / train_config['dataset_name']
if not data_dir.exists():
    zip_path = Path(SCRATCH) / (train_config["dataset_name"] + '.zip')
    if zip_path.exists():
        command = f'7z x {str(zip_path)} -o{LSCRATCH}'
    else:
        tar_path = Path(SCRATCH) / (train_config["dataset_name"] + '.tar')
        if not tar_path.exists():
            raise FileNotFoundError(
                f"{train_config['dataset_name']} not found in SCRATCH or LSCRATCH")
        command = f'tar -xf {str(tar_path)} -C {LSCRATCH}'
    print(f"Extracting training data to {data_dir} (will take a minute or so)")
    dds.run_command(command)
model = dds.Model(**train_config, base_dir=LSCRATCH)

if args.finetune:
    # Load earlier net, but keep final linear layer weights random
    # TODO: how specific is this to resnet?
    starting_net = args.finetune
    state_dict = torch.load('./models/' + starting_net + '.pth')
    state_dict['model'] = {
        k: v
        for k, v in state_dict['model'].items()
        if not (k.startswith('1.8') or k.startswith('1.9'))}
    errors = model.learner.model.load_state_dict(state_dict['model'], strict=False)
    del state_dict
    print("Messages from setting starting net weights: ", errors)
    model.learner.freeze()

result_name = model.train()

##
# Make a quick theta_E sigma_sub plot
##

# **This destroys learn.recorder!** Make sure you do not run save again!
results = model.predict_all()
plt.figure(figsize=(4,4))
plt.scatter(results['pred']['theta_E'],
            results['pred']['sigma_sub'],
            marker='.', edgecolor='none', s=2,)
plt.xlabel(r"$\theta_E$")
plt.ylabel(r"$\Sigma_\mathrm{sub}$")
plt.xlim(0.75, 1.6)
plt.ylim(0, 0.2)
plt.savefig('plots/' + result_name + '_qe_ssub.png', dpi=200, bbox_inches='tight')
