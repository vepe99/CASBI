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

if __name__ == '__main__':

    # N_subhalos = 2
    data = pd.read_parquet('../../../../data/dataframe/dataframe.parquet')
    data = rescale(data, mean_and_std_path='../../../../data/preprocess/mean_and_std.parquet', scale_observations=True, scale_parameter=True, inverse=True) 
    data =  data.drop(['gas_log10mass', 'a','redshift', 'mean_metallicity', 'std_metallicity','mean_FeMassFrac', 'std_FeMassFrac', 'mean_OMassFrac', 'std_OMassFrac'], axis=1)
    min_feh, max_feh = min(data['feh']), max(data['feh'])
    min_ofe, max_ofe = min(data['ofe']), max(data['ofe'])

    
    N_test = 1000
    def preprocess_testset(i):
        N_subhalos = np.random.randint(2, 100)
        galaxies = set(data['Galaxy_name'].drop_duplicates().sample(N_subhalos, random_state=i))
        # parameters =  data[data['Galaxy_name'].isin(galaxies)].drop(['feh', 'ofe', 'Galaxy_name'], axis=1).drop_duplicates().values.T
        # sorted_index = np.argsort(parameters[0], )[::-1]
        # parameters = (parameters[:,sorted_index]).reshape(-1)
        galaxy_data = data[data['Galaxy_name'].isin(galaxies)].values
        histogram_galaxy, _, _ = np.histogram2d(galaxy_data[:, 0], galaxy_data[:, 1], bins=64, range=[[min_feh, max_feh], [min_ofe, max_ofe]])
        sim_data =  np.expand_dims(np.log10(histogram_galaxy + 1e-6 +1), axis=0)
        return N_subhalos, sim_data, galaxies
    
    # Create a pool of workers
    with Pool() as pool:
        # Map the function to the data
        results = pool.map(preprocess_testset, range(N_test))
        
    # Unpack the results
    N_subhalos_test, x_test, galaxies_test = zip(*results)
    N_subhalos_test = np.array(N_subhalos_test).reshape((len(N_subhalos_test), 1))
    
    #take the first test set element as x_0 and theta_0    
    galaxies_0 = galaxies_test[0]
    data_to_plot_halos = data[data['Galaxy_name'].isin(galaxies_0)].to_parquet('./halos_0.parquet')
    N_subhalos_0 = N_subhalos_test[0]
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
        # parameters =  data[data['Galaxy_name'].isin(galaxies)].drop(['feh', 'ofe', 'Galaxy_name'], axis=1).drop_duplicates().values.T
        # sorted_index = np.argsort(parameters[0], )[::-1]
        # parameters = (parameters[:,sorted_index]).reshape(-1)
        galaxy_data = data[data['Galaxy_name'].isin(galaxies)].values
        histogram_galaxy, _, _ = np.histogram2d(galaxy_data[:, 0], galaxy_data[:, 1], bins=64, range=[[min_feh, max_feh], [min_ofe, max_ofe]])
        sim_data =  np.expand_dims(np.log10(histogram_galaxy + 1e-6 +1), axis=0)
        return N_subhalos, sim_data

    # Create a pool of workers
    with Pool() as pool:
        # Map the function to the data
        results = pool.map(process_sample, range(N))

    # Unpack the results
    N_subhalos, x = zip(*results)
    N_subhalos = np.array(N_subhalos).reshape((len(N_subhalos), 1)) #necessary for the sbi part that the shape is not just (len(N_subhalos),) but (len(N_subhalos),1)
    
    #save in .npy files, we remove the first element of the test set since it will be stored as x_0 and theta_0')
    path = '../../../../../../data/vgiusepp/'
    np.save(path+'x_test.npy', x_test[1:])
    np.save(path+'N_subhalos_test.npy', N_subhalos_test[1:])
    np.save(path+'x_0.npy', x_0)
    np.save(path+'N_subhalos_0.npy', N_subhalos_0)
    np.save(path+'x.npy', x)
    np.save(path+'N_subhalos.npy', N_subhalos)
    print('finish prepare the data')
    
    # reload all simulator examples as a dataloader
    all_loader = StaticNumpyLoader.from_config("./data.yaml")
    # all_loader = NumpyLoader(x=x, theta=N_subhalos, xobs =x_0, thetafid =N_subhalos_0)
    
    # class CustomDataset(Dataset):
    #     def __init__(self, images_path, labels_path, transform=None):
    #         self.images = torch.from_numpy(np.load(images_path)).float()
    #         self.labels = torch.from_numpy(np.load(labels_path)).float()
    #         self.transform = transform
            
    #         print(f"Image type: {type( self.images)}, shape: { self.images.shape}")
    #         print(f"Label type: {type(self.labels)}, shape: {self.labels.shape}")

    #     def __len__(self):
    #         return len(self.images)

    #     def __getitem__(self, idx):
    #         image = self.images[idx]
    #         label = self.labels[idx]
            
    #         if self.transform:
    #             image = self.transform(image)
                
    #         return image, label
        
    # batch_size = 1
    # path = '../../../../../../data/vgiusepp/'
    # train_dataset = CustomDataset(images_path=path+'x.npy', labels_path=path+'N_subhalos.npy')
    # test_dataset = CustomDataset(images_path=path+'x_test.npy', labels_path=path+'N_subhalos_test.npy') 

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # est_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



    # # Split the original training dataset into a new training dataset and a validation dataset
    # # Here, we use 80% of the images for training and 20% for validation
    # num_train = len(train_loader.dataset)
    # num_val = int(0.2 * num_train)
    # num_train = num_train - num_val
    # train_dataset, val_dataset = random_split(train_loader.dataset, [num_train, num_val])

    # # Create DataLoaders for the new training dataset and the validation dataset
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # all_loader = TorchLoader(train_loader, val_loader)
    

    # train a model to infer x -> theta. save it as toy/posterior.pkl
    runner = InferenceRunner.from_config(f"./training.yaml")
    runner(loader=all_loader)