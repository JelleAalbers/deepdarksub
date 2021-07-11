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
    '--batch_size', type=int, default=1024,
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
    '--dataset', default='dl_ss_npy',
    help='Dataset to use, must be a .zip in SCRATCH')
parser.add_argument(
    '--lr', default=0.1,
    help='Base learning rate to use')
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

SCRATCH = os.getenv('SCRATCH')    # Where training data zips are
LSCRATCH = os.getenv('LSCRATCH')  # Where to unzip (will make subfolder)
Path('./plots').mkdir(exist_ok=True)

train_config = dict(
    dataset_name = args.dataset,
    fit_parameters = (
        'main_deflector_parameters_center_x',
        'main_deflector_parameters_center_y',
        'main_deflector_parameters_gamma',
        'main_deflector_parameters_gamma1',
        'main_deflector_parameters_gamma2',
        'main_deflector_parameters_e1',
        'main_deflector_parameters_e2',
        'main_deflector_parameters_theta_E',
        'los_parameters_delta_los',
        'subhalo_parameters_sigma_sub'),
    uncertainty = args.uncertainty,
    augment_rotation = 'free',
    batch_size = args.batch_size,
    truncate_final = True,

    parameter_weights={
        'subhalo_parameters_sigma_sub': 10,
    },

    lr_schedule={
        'pct_start': args.pct_start
    },
    architecture_options = {
        'widen': args.widen
    },

    n_epochs = args.epochs,
    architecture = args.architecture,
    bn_final = True,
    base_lr = args.lr)

if train_config['uncertainty'] == 'correlated':
    # Haven't implemented weighting yet
    del train_config['parameter_weights']

data_dir = Path(LSCRATCH) / train_config['dataset_name']
if not data_dir.exists():
    print(f"Extracting training data to {data_dir} (will take a minute or so)")
    command = f'7z x {SCRATCH}/{train_config["dataset_name"]}.zip -o{LSCRATCH}'
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
