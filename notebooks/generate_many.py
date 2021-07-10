#!/usr/bin/env python
import argparse
import importlib
import os
from pathlib import Path
import subprocess
import sys
import textwrap


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='Submit sbatch jobs to generate a dataset with manada')
parser.add_argument(
    '--n_per_dir', type=int, default=10,
    help="Images per directory to simulate")
parser.add_argument(
    '--n_jobs', type=int, default=1,
    help='Directories / jobs to create')
parser.add_argument(
    '--manada_config', default='config_d_los_sigma_sub',
    help='Manada configuration name (without .py) to use')
parser.add_argument(
    '--dataset_name', default='test',
    help='Name of the dataset to create')
parser.add_argument(
    '--clear', action='store_true',
    help='Clear the temporary logs scripts dirs')
args = parser.parse_args()
    

def make_executable(path):
    """Make the file on path executable
    Stolen from some stackoverflow answer
    """
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2    # copy R bits to X
    os.chmod(path, mode)

# Could get it from deepdarksub.utils, 
# bbut imports take a long time on SDF...
def run_command(command, show_output=True):
    """Run command, return output and show in STDOUT"""
    # Is there no easier way??
    with subprocess.Popen(
            command.split(),
            bufsize=1,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT) as p:
        result = []
        for line in iter(p.stdout.readline, ''):
            line = line.rstrip()
            result.append(line)
            if show_output:
                print(line)
        return '\n'.join(result)

n_per_dir = args.n_per_dir
n_jobs = args.n_jobs

manada_config = args.manada_config
dataset_name = args.dataset_name

home_dir = os.path.expanduser('~')
logs_dir = home_dir + '/manada_generation_logs'
script_dir = home_dir + '/manada_generation_scripts'

# Use the current conda environment / python path
conda_env = os.getenv('CONDA_DEFAULT_ENV')
python_path = sys.executable

# Directory which contains manada's generate.py
manada_dir = str(Path(importlib.util.find_spec("manada").origin).parent)

out_path = Path(os.getenv('SCRATCH')) / dataset_name
if out_path.exists():
    raise FileExistsError(
        f"There is already a dataset at {out_path}. "
        "Choose a different dataset name, or delete the old dataset first.")
    
if args.clear:
    run_command(f"rm -r {logs_dir}")
    run_command(f"rm -r {script_dir}")

os.makedirs(logs_dir, exist_ok=True)
os.makedirs(script_dir, exist_ok=True)

conda_path = run_command('which conda', show_output=False)
conda_prefix = str(Path(conda_path).parent.parent)

for job_i in range(n_jobs):
    fn = script_dir + f'/script_{job_i}.sh'
    job_i = '%06d' % job_i
    
    script = textwrap.dedent(f"""\
    #!/bin/bash
    #SBATCH --job-name={dataset_name}_{job_i}
    #SBATCH --output={logs_dir}/output-%j.txt
    #SBATCH --error={logs_dir}/output-%j.txt
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=1
    #SBATCH --mem-per-cpu=1g
    #SBATCH --time=2:30:00

    export PATH={conda_prefix}/bin/:$PATH
    source {conda_prefix}/etc/profile.d/conda.sh
    conda activate {conda_env}

    cd {manada_dir}
    {python_path} generate.py Configs/{manada_config}.py {out_path}/{job_i} --n {n_per_dir}
    """)
    
    with open(fn, mode='w') as f:
        f.write(script)
    make_executable(fn)
    run_command(f"sbatch {fn}")
