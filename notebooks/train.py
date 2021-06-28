#!/usr/bin/env python

# Imports (and even just starting the kernel)
# takes a while on SDF, due to the network filesystem.
# Sometimes the kernel does not (re)start at all.
print("Imports started")
from pathlib import Path
import matplotlib.pyplot as plt
import deepdarksub as dds
print("Imports done")


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
    uncertainty = 'diagonal',
    augment_rotation = 'free',
    batch_size = 1024,
    
    parameter_weights={
        'subhalo_parameters_sigma_sub': 10,
    },
    
    lr_schedule={
        'pct_start': 0.1
    },
    
    n_epochs = 1,
    architecture = 'resnet18',
    bn_final = True,
    base_lr = 0.07)


data_dir = Path('/lscratch/jaalbers') / train_config['dataset_name']
if not data_dir.exists():
    print(f"Extracting training data to {data_dir} (will take a minute or so)")
    command = f'7z x /scratch/jaalbers/{train_config["dataset_name"]}.zip -o/lscratch/jaalbers'
    dds.run_command(command)
    
model = dds.Model(**train_config, base_dir='/lscratch/jaalbers')
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

Path('./plots').mkdir(exist_ok=True)
plt.savefig('plots/' + result_name + '_qe_ssub.png', dpi=200, bbox_inches='tight')
