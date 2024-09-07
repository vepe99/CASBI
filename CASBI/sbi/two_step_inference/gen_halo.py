"""
Generate galaxy halo from simulations and update and update the yaml file.
The inference is done in two steps:
1. Generate the galaxy halo to train the inference on the number of subhalos N -> gen_halo_Nsubhalos.py
2. Genereate the galaxy halo to train the inference on the parameters of the subhalos -? gen_halo.py

two supplementary function are used as utils:
-gen_onehalo: generate the galaxy halo for a given number of subhalos
"""


import os
import argparse
import yaml
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool

import torch 
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn

from ili.dataloaders import StaticNumpyLoader, TorchLoader, NumpyLoader
from ili.dataloaders import TorchLoader
from ili.inference import InferenceRunner
from ili.validation import ValidationRunner

from CASBI.utils.create_dataframe import rescale

"""
=========================================================
Generate the galaxy halo for the 2 step inference process
=========================================================
"""
def gen_onehalo(data, N_subhalos, train:bool, galaxies_test:np.array, min_feh, max_feh, min_ofe, max_ofe, ):
    """
    Function to genereate one halo for a given number of subhalos. If train=True, the function will check if each of the sets of subhalos is already present in the test set, and if so, it will generate a new set of subhalos.
    Args:
    data (pd.DataFrame): dataframe containing the data
    N_subhalos (int): number of subhalos
    train (bool): if True, check if the set of subhalos is already present in the test set
    galaxies_test (np.array): test set of galaxies
    min_feh (float): minimum value of feh
    max_feh (float): maximum value of feh
    min_ofe (float): minimum value of ofe
    max_ofe (float): maximum value of ofe
    random_state (int): random state to generate the subhalos
    
    Returns:
    N_subhalos (int): number of subhalos
    parameters (np.array): parameters of the subhalos
    sim_data (np.array): simulated data
    galaxies (np.array): list of galaxies
    
    """
    galaxies = set(data['Galaxy_name'].drop_duplicates().sample(N_subhalos, random_state=int(time.time())))
    if train: #if training check wheter or not those galaxy are present in the galaxy test set 
        while (any(set(galaxies) == galaxy_in_testset for galaxy_in_testset in galaxies_test)):
            print('matched galaxies, try again')
            print('galaxies', set(galaxies))
            print('test galaxies', galaxies_test)
            galaxies = data['Galaxy_name'].drop_duplicates().sample(N_subhalos, random_state=int(time.time()))
    parameters =  data[data['Galaxy_name'].isin(galaxies)].drop(['feh', 'ofe', 'Galaxy_name'], axis=1).drop_duplicates().values.T
    sorted_index = np.argsort(parameters[0], )[::-1] #orders the parameters in descending order of star mass
    parameters = (parameters[:,sorted_index]).reshape(-1)
    galaxy_data = data[data['Galaxy_name'].isin(galaxies)].values
    histogram_galaxy, _, _ = np.histogram2d(galaxy_data[:, 0], galaxy_data[:, 1], bins=64, range=[[min_feh, max_feh], [min_ofe, max_ofe]])
    sim_data =  np.expand_dims(np.log10(histogram_galaxy + 1e-6 +1), axis=0)
    return N_subhalos, parameters, sim_data, galaxies

def gen_halo_Nsubhalos(data:pd.DataFrame, output_dir:str, n_test:int, n_train:int, max_subhalos:int=100, ):
    """
    Generate the galaxy halo to train the inference on the number of subhalos N.
    Args:
    data (pd.DataFrame): dataframe with observation and parameters already rescaled to physical values
    output_dir (str): path to the output directory where to store the data, important to separete it from the gen_halo function output directory
    n_test (int): number of test samples, it will be used to generate n_test samples for each number of subhalos from 2 to max_subhalos
    n_train (int): number of training samples, it will be used to generate n_train samples for each number of subhalos from 2 to max_subhalos
    max_subhalos (int): maximum number of subhalos
    
    Returns:
    N_subhalos_test (np.array): number of subhalos for the test set
    parameters_test (np.array): parameters of the subhalos for the test set
    x_test (np.array): simulated data for the test set
    galaxies_test (np.array): list of galaxies for the test set
    N_subhalos_0 (np.array): number of subhalos for the first test set element
    x_0 (np.array): simulated data for the first test set element
    N_subhalos (np.array): number of subhalos for the training set
    parameters (np.array): parameters of the subhalos for the training set
    x (np.array): simulated data for the training set
    galaxies_training (np.array): list of galaxies for the training set
    """

    # data = pd.read_parquet(data_file)
    # data = rescale(data, mean_and_std_path=rescale_file, scale_observations=True, scale_parameters=True, inverse=True) 
    # data =  data.drop(['gas_log10mass', 'a','redshift', 'mean_metallicity', 'std_metallicity','mean_FeMassFrac', 'std_FeMassFrac', 'mean_OMassFrac', 'std_OMassFrac'], axis=1)
    min_feh, max_feh = min(data['feh']), max(data['feh'])
    min_ofe, max_ofe = min(data['ofe']), max(data['ofe'])

    arr = np.concatenate([np.repeat(i, n_test) for i in range(2, max_subhalos)]) #generate n_test samples for each number of subhalos from 2 to max_subhalos
    np.random.shuffle(arr)
    # Create a pool of workers
    with Pool() as pool:
        # Map the function to the data
        results = pool.starmap(gen_onehalo, [(data, n_subhalos, 0, None, min_feh, max_feh, min_ofe, max_ofe, ) for e, n_subhalos in enumerate(arr)]) #n_subhalos is passed also as random state to generate the subhalos
        
    # Unpack the results
    N_subhalos_test, parameters_test, x_test, galaxies_test = zip(*results)
    N_subhalos_test = np.array(N_subhalos_test).reshape((len(N_subhalos_test), 1))  

    #take the first test set element as x_0 and theta_0    
    galaxies_0 = galaxies_test[0]
    # data_to_plot_halos = data[data['Galaxy_name'].isin(galaxies_0)].to_parquet('./halos_0.parquet')
    N_subhalos_0 = N_subhalos_test[0]
    x_0 =  x_test[0]
        
    arr = np.concatenate([np.repeat(i, n_train) for i in range(2, max_subhalos)]) #generate n_train samples for each number of subhalos from 2 to max_subhalos
    np.random.shuffle(arr)
    # Create a pool of workers
    with Pool() as pool:
        # Map the function to the data
        results = pool.starmap(gen_onehalo, [(data, n_subhalos, 1, galaxies_test, min_feh, max_feh, min_ofe, max_ofe, ) for e, n_subhalos in enumerate(arr)]) #n_subhalos is passed also as random state to generate the subhalos

    # Unpack the results
    N_subhalos, parameters, x, galaxies_training = zip(*results)
    N_subhalos = np.array(N_subhalos).reshape((len(N_subhalos), 1))

    #save in .npy files, we remove the first element of the test set since it will be stored as x_0 and theta_0')
    np.save(os.path.join(output_dir,'x_test.npy'), x_test[1:])
    np.save(os.path.join(output_dir,'N_subhalos_test.npy'), N_subhalos_test[1:])
    np.save(os.path.join(output_dir, 'galaxy_test.npy'), galaxies_test[1:])
    np.save(os.path.join(output_dir,'x_0.npy'), x_0)
    np.save(os.path.join(output_dir,'N_subhalos_0.npy'), N_subhalos_0)
    np.save(os.path.join(output_dir,'parameters_0.npy'), parameters_test[0])
    np.save(os.path.join(output_dir,'galaxy_0.npy'), galaxies_0)
    np.save(os.path.join(output_dir,'x.npy'), x)
    np.save(os.path.join(output_dir,'N_subhalos.npy'), N_subhalos)
    np.save(os.path.join(output_dir, 'galaxy_training.npy'), galaxies_training)
    print('finish prepare the data')
    
    return N_subhalos_test, parameters_test, x_test, galaxies_test, N_subhalos_0, x_0, N_subhalos, parameters, x, galaxies_training
    
def gen_halo(data:pd.DataFrame, output_dir:str, training_yaml:str, n_test:int, n_train:int, N_subhalos:int):
    """
    Generate the galaxy halo to train the inference on the parameters of the subhalos.
    Args:
    data (pd.DataFrame): dataframe with observation and parameters already rescaled to physical values
    output_dir (str): path to the output directory where to store the data, important to separete it from the gen_halo_Nsubhalos function output directory
    n_test (int): number of test samples 
    n_train (int): number of training samples
    N_subhalos (int): number of subhalos, should be output of the inference part on the number of subhalos
    
    Returns:
    N_subhalos_test (np.array): number of subhalos for the test set
    theta_test (np.array): parameters of the subhalos for the test set
    x_test (np.array): simulated data for the test set
    galaxies_test (np.array): list of galaxies for the test set
    N_subhalos_0 (np.array): number of subhalos for the first test set element
    theta_0 (np.array): parameters of the subhalos for the first test set element
    N_subhalos (np.array): number of subhalos for the training set
    theta (np.array): parameters of the subhalos for the training set
    x (np.array): simulated data for the training set
    galaxies_training (np.array): list of galaxies for the training set
    """
    # data = pd.read_parquet(data_file)
    # data = rescale(data, mean_and_std_path=rescale_file, scale_observations=True, scale_parameter=True, inverse=True) 
    # data =  data.drop(['gas_log10mass', 'a','redshift', 'mean_metallicity', 'std_metallicity','mean_FeMassFrac', 'std_FeMassFrac', 'mean_OMassFrac', 'std_OMassFrac'], axis=1)
    print('running gen halo')
    min_feh, max_feh = min(data['feh']), max(data['feh'])
    min_ofe, max_ofe = min(data['ofe']), max(data['ofe'])
    conditions = data[data.columns.difference(['feh', 'ofe', 'Galaxy_name'], sort=False)].drop_duplicates()
    
    ####update the prior to accomodate for the right number of subhalos
    minimum_theta = [conditions[col].values.min() for col in conditions.columns]   
    maximum_theta = [conditions[col].values.max() for col in conditions.columns]       
    minimum_theta = np.array(minimum_theta)
    maximum_theta = np.array(maximum_theta)

    repeat_minimum_theta = np.repeat(minimum_theta, N_subhalos)
    repeat_maximum_theta = np.repeat(maximum_theta, N_subhalos) 

    with open(training_yaml, 'r') as file:
        yaml_file = yaml.safe_load(file)

    repeat_minimum_theta = repeat_minimum_theta.tolist()
    repeat_maximum_theta = repeat_maximum_theta.tolist()
    # Update the value
    yaml_file['prior']['args']['low'] = repeat_minimum_theta
    yaml_file['prior']['args']['high'] = repeat_maximum_theta

    # Write the data back to the file
    with open(training_yaml, 'w') as file:
        yaml.safe_dump(yaml_file, file)
            
    print('write the right prior in the training.yaml file')
    
    arr = np.array([N_subhalos for i in range(n_test)])
    # Create a pool of workers
    with Pool() as pool:
        # Map the function to the data
        results = pool.starmap(gen_onehalo, [(data, N_subhalos, 0, None, min_feh, max_feh, min_ofe, max_ofe,  ) for i in range(n_test)]) #the index i is passed as random state to generate the subhalos
        
    # Unpack the results
    N_subhalos_test, theta_test, x_test, galaxies_test = zip(*results)
    
    #take the first test set element as x_0 and theta_0    
    galaxies_0 = galaxies_test[0]
    data_to_plot_halos = data[data['Galaxy_name'].isin(galaxies_0)].to_parquet('./halos_0.parquet')
    theta_0 =  theta_test[0]
    x_0 =  x_test[0]

    arr = np.array([N_subhalos for i in range(n_train)])
    # Create a pool of workers
    with Pool() as pool:
        # Map the function to the data
        results = pool.starmap(gen_onehalo, [(data, N_subhalos, 1, galaxies_test, min_feh, max_feh, min_ofe, max_ofe) for i in range(n_train)]) #the index i is passed as random state to generate the subhalos
    # Unpack the results
    N_subhalos, theta, x, galaxies = zip(*results)
    
    #save in .npy files, we remove the first element of the test set since it will be stored as x_0 and theta_0
    np.save(os.path.join(output_dir,'./x_test.npy'), x_test[1:])
    np.save(os.path.join(output_dir, './theta_test.npy'), theta_test[1:])
    np.save(os.path.join(output_dir, './galaxy_test.npy'), galaxies_test[1:])
    np.save(os.path.join(output_dir,'./x_0.npy'), x_0)
    np.save(os.path.join(output_dir,'./theta_0.npy'), theta_0)
    np.save(os.path.join(output_dir,'./x.npy'), x)
    np.save(os.path.join(output_dir,'./theta.npy'), theta)
    np.save(os.path.join(output_dir, './galaxy_training.npy'), galaxies)
    print('finish prepare the data')
    return N_subhalos_test, theta_test, x_test, galaxies_test, N_subhalos, theta, x, galaxies



"""
==============
histogram data
==============
"""
import random

def gen_onehalo_hist(data_dir, N_subhalos, train:bool, galaxies_test:np.array, inference_galaxy:list ):
    """
    """
    random.seed(int(time.time()))
    galaxies_removed_inference = [g for g in os.listdir(data_dir) if g not in inference_galaxy] #remove the galaxy used for inference
    galaxies = set(random.sample(galaxies_removed_inference, N_subhalos))
    if train: #if training check wheter or not those galaxy are present in the galaxy test set 
        while (any(set(galaxies) == galaxy_in_testset for galaxy_in_testset in galaxies_test)):
            print('matched galaxies, try again')
            print('galaxies', set(galaxies))
            print('test galaxies', galaxies_test)
            galaxies = set(random.sample(os.listdir(data_dir), N_subhalos))
    parameters =  np.array([[np.load(os.path.join(data_dir, g))['star_log10mass'], np.load(os.path.join(data_dir, g))['dm_log10mass'], np.load(os.path.join(data_dir, g))['infall_time']]  for g in galaxies ]).T
    sorted_index = np.argsort(parameters[0], )[::-1] #orders the parameters in descending order of star mass
    parameters = (parameters[:,sorted_index]).reshape(-1)
    # sim_data =  np.log10(np.sum(np.array([np.load(os.path.join(data_dir, g))['observables'] for g in galaxies]), axis=0)[np.newaxis, :] + 1+1e-6)
    sim_data =  np.sum(np.array([np.load(os.path.join(data_dir, g))['observables'] for g in galaxies]), axis=0)[np.newaxis, :] 
    return N_subhalos, parameters, sim_data, galaxies


def gen_halo_Nsubhalos_hist(data_dir:pd.DataFrame, output_dir:str, n_test:int, n_train:int, inference_galaxy:str, max_subhalos:int=100, ):
    """
    """
    
    arr = np.concatenate([np.repeat(i, n_test) for i in range(2, max_subhalos)]) #generate n_test samples for each number of subhalos from 2 to max_subhalos
    np.random.shuffle(arr)
    # Create a pool of workers
    with Pool() as pool:
        # Map the function to the data
        results = pool.starmap(gen_onehalo_hist, [(data_dir, n_subhalos, 0, None, inference_galaxy) for n_subhalos in arr]) #n_subhalos is passed also as random state to generate the subhalos
        
    # Unpack the results
    N_subhalos_test, parameters_test, x_test, galaxies_test = zip(*results)
    N_subhalos_test = np.array(N_subhalos_test).reshape((len(N_subhalos_test), 1))  #needed for sbi input shape 

    #take the first test set element as x_0 and theta_0    
    galaxies_0 = galaxies_test[0]
    # data_to_plot_halos = data[data['Galaxy_name'].isin(galaxies_0)].to_parquet('./halos_0.parquet')
    N_subhalos_0 = N_subhalos_test[0]
    x_0 =  x_test[0]
        
    arr = np.concatenate([np.repeat(i, n_train) for i in range(2, max_subhalos)]) #generate n_train samples for each number of subhalos from 2 to max_subhalos
    np.random.shuffle(arr)
    # Create a pool of workers
    with Pool() as pool:
        # Map the function to the data
        results = pool.starmap(gen_onehalo_hist, [(data_dir, n_subhalos, 1, galaxies_test, inference_galaxy) for n_subhalos in arr]) #n_subhalos is passed also as random state to generate the subhalos

    # Unpack the results
    N_subhalos, parameters, x, galaxies_training = zip(*results)
    N_subhalos = np.array(N_subhalos).reshape((len(N_subhalos), 1))

    #save in .npy files, we remove the first element of the test set since it will be stored as x_0 and theta_0')
    np.save(os.path.join(output_dir,'x_test.npy'), x_test[1:])
    np.save(os.path.join(output_dir,'N_subhalos_test.npy'), N_subhalos_test[1:])
    np.save(os.path.join(output_dir, 'galaxy_test.npy'), galaxies_test[1:])
    np.save(os.path.join(output_dir,'x_0.npy'), x_0)
    np.save(os.path.join(output_dir,'N_subhalos_0.npy'), N_subhalos_0)
    np.save(os.path.join(output_dir,'parameters_0.npy'), parameters_test[0])
    np.save(os.path.join(output_dir,'galaxy_0.npy'), galaxies_0)
    np.save(os.path.join(output_dir,'x.npy'), x)
    np.save(os.path.join(output_dir,'N_subhalos.npy'), N_subhalos)
    np.save(os.path.join(output_dir, 'galaxy_training.npy'), galaxies_training)
    print('finish prepare the data')
    
    return N_subhalos_test, parameters_test, x_test, galaxies_test, N_subhalos_0, x_0, N_subhalos, parameters, x, galaxies_training

def gen_halo_hist(data_dir:pd.DataFrame, output_dir:str, training_yaml:str, n_test:int, n_train:int, N_subhalos:int, inference_galaxy,  prior_limits):
    """
    """
    print('running gen halo')
    
    ####update the prior to accomodate for the right number of subhalos
    minimum_theta = [prior_limits[col].values.min() for col in prior_limits.columns]   
    maximum_theta = [prior_limits[col].values.max() for col in prior_limits.columns]       
    minimum_theta = np.array(minimum_theta)
    maximum_theta = np.array(maximum_theta)

    repeat_minimum_theta = np.repeat(minimum_theta, N_subhalos)
    repeat_maximum_theta = np.repeat(maximum_theta, N_subhalos) 

    with open(training_yaml, 'r') as file:
        yaml_file = yaml.safe_load(file)

    repeat_minimum_theta = repeat_minimum_theta.tolist()
    repeat_maximum_theta = repeat_maximum_theta.tolist()
    # Update the value
    yaml_file['prior']['args']['low'] = repeat_minimum_theta
    yaml_file['prior']['args']['high'] = repeat_maximum_theta

    # Write the data back to the file
    with open(training_yaml, 'w') as file:
        yaml.safe_dump(yaml_file, file)
            
    print('write the right prior in the training.yaml file')
    
    # Create a pool of workers
    with Pool() as pool:
        # Map the function to the data
        results = pool.starmap(gen_onehalo_hist, [(data_dir, N_subhalos, 0, None, inference_galaxy ) for i in range(n_test)]) #the index i is passed as random state to generate the subhalos
        
    # Unpack the results
    N_subhalos_test, theta_test, x_test, galaxies_test = zip(*results)
    
    #take the first test set element as x_0 and theta_0    
    galaxies_0 = galaxies_test[0]
    theta_0 =  theta_test[0]
    x_0 =  x_test[0]

    # Create a pool of workers
    with Pool() as pool:
        # Map the function to the data
        results = pool.starmap(gen_onehalo_hist, [(data_dir, N_subhalos, 1, galaxies_test, inference_galaxy) for i in range(n_train)]) #the index i is passed as random state to generate the subhalos
    # Unpack the results
    N_subhalos, theta, x, galaxies = zip(*results)
    
    #save in .npy files, we remove the first element of the test set since it will be stored as x_0 and theta_0
    np.save(os.path.join(output_dir,'./x_test.npy'), x_test[1:])
    np.save(os.path.join(output_dir, './theta_test.npy'), theta_test[1:])
    np.save(os.path.join(output_dir, './galaxy_test.npy'), galaxies_test[1:])
    np.save(os.path.join(output_dir,'./x_0.npy'), x_0)
    np.save(os.path.join(output_dir,'./theta_0.npy'), theta_0)
    np.save(os.path.join(output_dir,'./galaxies_0.npy'), galaxies_0)
    np.save(os.path.join(output_dir,'./x.npy'), x)
    np.save(os.path.join(output_dir,'./theta.npy'), theta)
    np.save(os.path.join(output_dir, './galaxy_training.npy'), galaxies)
    print('finish prepare the data')
    return N_subhalos_test, theta_test, x_test, galaxies_test, N_subhalos, theta, x, galaxies