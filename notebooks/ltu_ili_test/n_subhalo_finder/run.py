import os
import argparse
import yaml
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool

from ili.dataloaders import StaticNumpyLoader
from ili.inference import InferenceRunner
from ili.validation import ValidationRunner

from CASBI.utils.create_dataframe import rescale

if __name__ == '__main__':

    # N_subhalos = 2
    data = pd.read_parquet('../../../../data/dataframe/dataframe.parquet')
    data = rescale(data, mean_and_std_path='../../../../data/preprocess/mean_and_std.parquet', scale_observations=True, scale_parameter=True, inverse=True) 
    data =  data.drop(['gas_log10mass', 'a','redshift', 'mean_metallicity', 'std_metallicity','mean_FeMassFrac', 'std_FeMassFrac', 'mean_OMassFrac', 'std_OMassFrac'], axis=1)
    min_feh, max_feh = min(data['feh']), max(data['feh'])
    min_ofe, max_ofe = min(data['ofe']), max(data['ofe'])

    
    N_test = 1_000
    def preprocess_testset(i):
        N_subhalos = np.random.randint(2, 100)
        galaxies = set(data['Galaxy_name'].drop_duplicates().sample(N_subhalos, random_state=i))
        parameters =  data[data['Galaxy_name'].isin(galaxies)].drop(['feh', 'ofe', 'Galaxy_name'], axis=1).drop_duplicates().values.T
        sorted_index = np.argsort(parameters[0], )[::-1]
        parameters = (parameters[:,sorted_index]).reshape(-1)
        galaxy_data = data[data['Galaxy_name'].isin(galaxies)].values
        histogram_galaxy, _, _ = np.histogram2d(galaxy_data[:, 0], galaxy_data[:, 1], bins=64, range=[[min_feh, max_feh], [min_ofe, max_ofe]])
        sim_data =  np.expand_dims(np.log10(histogram_galaxy + 1e-6 +1), axis=0)
        return N_subhalos, parameters, sim_data, galaxies
    
    # Create a pool of workers
    with Pool() as pool:
        # Map the function to the data
        results = pool.map(preprocess_testset, range(N_test))
        
    # Unpack the results
    N_subhalos_test, theta_test, x_test, galaxies_test = zip(*results)
    
    #take the first test set element as x_0 and theta_0    
    galaxies_0 = galaxies_test[0]
    data_to_plot_halos = data[data['Galaxy_name'].isin(galaxies_0)].to_parquet('./halos_0.parquet')
    N_subhalos_0 = N_subhalos_test[0]
    theta_0 =  theta_test[0]
    x_0 =  x_test[0]

    N = 50_000
    def process_sample(i):
        N_subhalos = np.random.randint(2, 100)
        galaxies = data['Galaxy_name'].drop_duplicates().sample(N_subhalos, random_state=i+int(time.time()))
        while (any(set(galaxies) == galaxy_in_testset for galaxy_in_testset in galaxies_test)):
            print('matched galaxies, try again')
            print('galaxies', set(galaxies))
            print('test galaxies', galaxies_test)
            galaxies = data['Galaxy_name'].drop_duplicates().sample(N_subhalos, random_state=i)
        parameters =  data[data['Galaxy_name'].isin(galaxies)].drop(['feh', 'ofe', 'Galaxy_name'], axis=1).drop_duplicates().values.T
        sorted_index = np.argsort(parameters[0], )[::-1]
        parameters = (parameters[:,sorted_index]).reshape(-1)
        galaxy_data = data[data['Galaxy_name'].isin(galaxies)].values
        histogram_galaxy, _, _ = np.histogram2d(galaxy_data[:, 0], galaxy_data[:, 1], bins=64, range=[[min_feh, max_feh], [min_ofe, max_ofe]])
        sim_data =  np.expand_dims(np.log10(histogram_galaxy + 1e-6 +1), axis=0)
        return parameters, sim_data

    # Create a pool of workers
    with Pool() as pool:
        # Map the function to the data
        results = pool.map(process_sample, range(N))

    # Unpack the results
    theta, x = zip(*results)
    
    #save in .npy files, we remove the first element of the test set since it will be stored as x_0 and theta_0
    np.save('./x_test.npy', x_test[1:])
    np.save('./theta_test.npy', theta_test[1:])
    np.save('./x_0.npy', x_0)
    np.save('./theta_0.npy', theta_0)
    np.save('./x.npy', x)
    np.save('./theta.npy', theta)
    print('finish prepare the data')

    # Convert to numpy arrays
    # theta = torch.from_numpy(np.array(theta)).float()
    # x = torch.from_numpy(np.array(x)).float()
    
    
    # reload all simulator examples as a dataloader
    all_loader = StaticNumpyLoader.from_config("./data.yaml")

    # train a model to infer x -> theta. save it as toy/posterior.pkl
    runner = InferenceRunner.from_config(
        f"./training.yaml")
    runner(loader=all_loader)