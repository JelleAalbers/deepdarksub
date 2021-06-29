#!/usr/bin/env python


import argparse

parser = argparse.ArgumentParser(description='Train a substructure predicting neural net')
parser.add_argument(
    '--finetune', metavar='netname', default=None,
    help="Train a head onto another net's frozen body")
parser.add_argument(
    '--epochs', metavar='N', type=int, default=1,
    help="Number of training epochs (default 1)")
parser.add_argument(
    '--uncertainty', default='diagonal',
    help='Kind of uncertainty the net should output: diagonal or correlated')
args = parser.parse_args()


# Imports take a while on SDF due to the network filesystem.
print("Imports started")
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import deepdarksub as dds
print("Imports done")

Path('./plots').mkdir(exist_ok=True)

train_config = dict(
    dataset_name = 'dl_ss_npy',
    fit_parameters = (
        'main_deflector_parameters_theta_E',
        'subhalo_parameters_sigma_sub',
        'los_parameters_delta_los',
        'main_deflector_parameters_center_x',
        'main_deflector_parameters_center_y',
        'main_deflector_parameters_gamma',
        'main_deflector_parameters_gamma1',
        'main_deflector_parameters_gamma2',
        'main_deflector_parameters_e1',
        'main_deflector_parameters_e2'),
    uncertainty = args.uncertainty,
    augment_rotation = 'free',
    batch_size = 1024,
    
    parameter_weights={
        'subhalo_parameters_sigma_sub': 10,
    },
    
    # lr_schedule={
    #     'pct_start': 0.1
    # },
    
    n_epochs = args.epochs,
    architecture = 'xse_resnet18',
    bn_final = True,
    base_lr = 0.1)

if train_config['uncertainty'] == 'correlated':
    # Haven't implemented weighting yet
    del train_config['parameter_weights']

data_dir = Path('/lscratch/jaalbers') / train_config['dataset_name']
if not data_dir.exists():
    print(f"Extracting training data to {data_dir} (will take a minute or so)")
    command = f'7z x /scratch/jaalbers/{train_config["dataset_name"]}.zip -o/lscratch/jaalbers'
    dds.run_command(command)
    
model = dds.Model(**train_config, base_dir='/lscratch/jaalbers')

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
