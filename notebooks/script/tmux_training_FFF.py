# from CASBI.generator.fff.fff_model import *
from CASBI.generator.fff.fff_model import FreeFormFlow
from CASBI.generator.create_test_val_set import load_train_objs

import numpy as np
import pandas as pd

import torch
import os
import yaml
import argparse


# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, required=True, help='Device index for training')
args = parser.parse_args()

def create_directory_and_save_files(base_dir, hyperparameters):
    # Get the names of all directories in the base directory
    dir_names = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]

    # Find the highest existing test number
    highest_test_num = 0
    for name in dir_names:
        if name.startswith('test_'):
            try:
                test_num = int(name.split('_')[1])
                highest_test_num = max(highest_test_num, test_num)
            except ValueError:
                pass  # Ignore directory names that don't have a number after 'test_'

    # Create a new directory with the next test number
    new_dir_name = 'test_{}'.format(highest_test_num + 1)
    new_dir_path = os.path.join(base_dir, new_dir_name)
    os.makedirs(new_dir_path, exist_ok=True)

    # Save the hyperparameters to a .yaml file
    with open(os.path.join(new_dir_path, 'hyperparameters.yaml'), 'w') as f:
        yaml.dump(hyperparameters, f)

    # Copy the snapshot.pth and tensorboard files to the new directory
    new_tensorboard_path = os.path.join(new_dir_path, "tensorboard")


    # Return the paths to the new snapshot and tensorboard files
    return new_dir_path, new_tensorboard_path


hyperparameters = {
    'N_val':200,
    'N_test':30, 
    'dim': 2,
    'cond_dim': 4,
    'hidden_dim': 128,
    'latent_dim': 2,
    'n_SC_layer': 8,
    'beta': 100**2,
    'device': 'cuda:{}'.format(args.device),
    'n_epochs': 200,
    'batch_size': 1024,
    'full_dataset': False,
    'independent_conditioning': True,
}

new_dir_path, new_tensorboard_path = create_directory_and_save_files('/export/home/vgiusepp/data/tuning', hyperparameters)
if hyperparameters['full_dataset']:
    train_set, val_set, test_set = load_train_objs('../../data/full_dataframe/dataframe/dataframe.parquet', N_val=hyperparameters['N_val'], N_test=hyperparameters['N_test'])
else:
    train_set, val_set, test_set = load_train_objs('../../data/dataframe/dataframe_2.parquet', N_val=hyperparameters['N_val'], N_test=hyperparameters['N_test'], sets_path=new_dir_path)

test_set = test_set[train_set.columns.difference(['Galaxy_name'], sort=False)]
if hyperparameters['independent_conditioning']:
    train_set = train_set[train_set.columns.difference([ 'a','redshift', 'mean_metallicity', 'std_metallicity', 'mean_FeMassFrac', 'std_FeMassFrac', 'mean_OMassFrac', 'std_OMassFrac', 'Galaxy_name'], sort=False)]
    val_set = val_set[train_set.columns.difference(['a','redshift', 'mean_metallicity', 'std_metallicity', 'mean_FeMassFrac', 'std_FeMassFrac', 'mean_OMassFrac', 'std_OMassFrac', 'Galaxy_name'], sort=False)]
else:
    train_set = train_set[train_set.columns.difference(['a','redshift', 'mean_FeMassFrac', 'std_FeMassFrac', 'mean_OMassFrac', 'std_OMassFrac', 'Galaxy_name'], sort=False)]
    val_set = val_set[train_set.columns.difference(['a','redshift', 'mean_FeMassFrac', 'std_FeMassFrac', 'mean_OMassFrac', 'std_OMassFrac', 'Galaxy_name'], sort=False)]
test_set = torch.from_numpy(np.array(test_set.values, dtype=float))
val_set = torch.from_numpy(np.array(val_set.values, dtype=float))
train_set = torch.from_numpy(np.array(train_set.values, dtype=float))


Flow = FreeFormFlow(dim = hyperparameters['dim'], 
                    cond_dim = hyperparameters['cond_dim'],
                    hidden_dim = hyperparameters['hidden_dim'],
                    latent_dim = hyperparameters['latent_dim'],
                    n_SC_layer = hyperparameters['n_SC_layer'],
                    beta = hyperparameters['beta'],
                    device = hyperparameters['device'],
                    )

Flow.train_model(n_epochs = hyperparameters['n_epochs'],
                 batch_size=hyperparameters['batch_size'],
                 optimizer=torch.optim.Adam(Flow.parameters(), lr=2e-4),
                 train_set=train_set,
                 val_set=val_set,
                 snapshot_path=new_dir_path, runs_path=new_tensorboard_path,
                 )