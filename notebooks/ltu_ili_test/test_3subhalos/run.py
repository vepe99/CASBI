import os
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool

from ili.dataloaders import StaticNumpyLoader
from ili.inference import InferenceRunner
from ili.validation import ValidationRunner

from CASBI.utils.create_dataframe import rescale

if __name__ == '__main__':

    N_subhalos = 3
    data = pd.read_parquet('../../../../data/dataframe/dataframe.parquet')
    data = rescale(data, mean_and_std_path='../../../../data/preprocess/mean_and_std.parquet', scale_observations=True, scale_parameter=True, inverse=True) 
    data =  data.drop(['gas_log10mass', 'a','redshift', 'mean_metallicity', 'std_metallicity','mean_FeMassFrac', 'std_FeMassFrac', 'mean_OMassFrac', 'std_OMassFrac'], axis=1)
    min_feh, max_feh = min(data['feh']), max(data['feh'])
    min_ofe, max_ofe = min(data['ofe']), max(data['ofe'])
    galaxies_0 = data['Galaxy_name'].sample(N_subhalos)
    data_to_plot_halos = data[data['Galaxy_name'].isin(galaxies_0)].to_parquet('./halos_0.parquet')
    theta_0 =  data[data['Galaxy_name'].isin(galaxies_0)].drop(['feh', 'ofe', 'Galaxy_name'], axis=1).drop_duplicates().values.T.reshape(-1)
    galaxy_data = data[data['Galaxy_name'].isin(galaxies_0)].values
    histogram_galaxy, _, _ = np.histogram2d(galaxy_data[:, 0], galaxy_data[:, 1], bins=64, range=[[min_feh, max_feh], [min_ofe, max_ofe]])
    x_0 =  np.expand_dims(np.log10(histogram_galaxy + 1e-6 +1), axis=0)

    N = 10_000
    def process_sample(i):
        galaxies = data['Galaxy_name'].drop_duplicates().sample(N_subhalos, random_state=i)
        while set(galaxies) == set(galaxies_0):
            print('matched galaxies, try again')
            galaxies = data['Galaxy_name'].drop_duplicates().sample(N_subhalos, random_state=i)
        parameters =  data[data['Galaxy_name'].isin(galaxies)].drop(['feh', 'ofe', 'Galaxy_name'], axis=1).drop_duplicates().values.T.reshape(-1)
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
    
    #save in .npy files 
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