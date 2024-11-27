import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.cm as cm

    
from scipy.stats import gaussian_kde


import pandas as pd
from multiprocessing import cpu_count
from multiprocessing import Pool

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn

import ili
from ili.dataloaders import NumpyLoader, TorchLoader
from ili.inference import InferenceRunner
from ili.validation.metrics import PosteriorCoverage, PlotSinglePosterior
from CASBI.utils.CNN import ConvNet

"""
==========
Inference
==========
In this script we will perform inference on the model trained in the training script. There are also the plot functions to evaluate
on a single test set object and a function to evaluate on the whole test set. Ltu-ili allow for a flexible yaml file interface,
we provide the function to create the yaml file.

"""
   
def train_inference(x:torch.Tensor, 
                    theta:torch.Tensor,
                    validation_fraction:float=0.2,
                    output_dir:str='./', 
                    device:str='cuda',
                    N_nets=4, 
                    hidden_feature:int=100, 
                    num_transforms:int=20, 
                    model:str='nsf', 
                    embedding_net:str = ConvNet(output_dim=32), 
                    minimum_theta:list=[3.5, -2.],
                    maximum_theta:list=[10, 1.15],
                    batch_size:int=1024,
                    learning_rate:float=0.00001,
                    stop_after_epochs:int=20):
    """
    Train a ltu-ili ensable model on (x, theta) couples. The training data is split into training and validation data. The model is saved in the output_dir directory.
    
    Parameters
    ----------
    x : torch.Tensor
        The observation data.
    theta : torch.Tensor
        The parameter data.
    validation_fraction : float, optional
        The fraction of the data to use for validation, by default 0.2
    output_dir : str, optional
        The directory where the model is saved, by default './'
    device : str, optional
        The device to use, by default 'cuda'
    N_nets : int, optional
        The number of nets to ensable, by default 4
    hidden_feature : int, optional
        The number of hidden features of the model, by default 100
    num_transforms : int, optional
        The number of transforms of the model, by default 10
    model : str, optional
        The model to use, by default 'nsf'
    embedding_net : str, optional
        The embedding network to use, by default 'CNN'. The CNN.py file should be in the same directory as the training script.
    output_dim : int, optional
        The output dimension of the embedding network, by default 32
    minimum_theta : list, optional
        The minimum value of the theta parameters, by default [3.5, -2.]
    maximum_theta : list, optional
        The maximum value of the theta parameters, by default [10, 1.15]
    batch_size : int, optional
        The batch size, by default 1024
    learning_rate : float, optional 
        The learning rate, by default 0.00001
    stop_after_epochs : int, optional
        The number of epochs to train, by default 20
        
    Returns
    -------
    posterior_ensemble : ili.posterior.PosteriorEnsemble
        The posterior ensemble model.
    summaries : dict
        The summaries of the training process.
    """
    
    #traing arguments
    train_args = {
    'training_batch_size': batch_size,
    'learning_rate': learning_rate,
    'stop_after_epochs': stop_after_epochs,
    }
    
    #ensable model
    runner = InferenceRunner.load(
        backend ='lampe',
        engine ='NPE',
        prior = ili.utils.Uniform(low=minimum_theta, high=maximum_theta, device=device),
        nets = [ili.utils.load_nde_lampe(model=model, hidden_features=hidden_feature, num_transforms=num_transforms,
                    embedding_net=embedding_net.to(device), x_normalize=False, device=device) for j in range(N_nets)],
        device=device,
        train_args=train_args,
        proposal=None,
        out_dir=output_dir,
    )
    
    # Assert if the data are PyTorch tensors
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    if not torch.is_tensor(theta):
        theta = torch.tensor(theta)
    
    
    #split validataion and training data
    data_size = len(x)
    print(data_size)    
    indices = np.random.permutation(data_size)

    # Decide on the split size, for example 80% for training and 20% for validation
    split_idx = int(data_size * (1-validation_fraction))

    # Split the indices
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]

    # Create the data splits
    train_data, train_targets = x[train_indices].float(), theta[train_indices].float()
    val_data, val_targets = x[val_indices].float(), theta[val_indices].float(),

    # Now you can create your DataLoaders
    train_loader = torch.utils.data.DataLoader(CustomDataset(train_data.to(device), train_targets.to(device),), shuffle=True, batch_size=2024)
    val_loader = torch.utils.data.DataLoader(CustomDataset(val_data.to(device), val_targets.to(device),), shuffle=False, batch_size=2024)
    # test_loader = DataLoader(test_dataset,  shuffle=False)

    loader = TorchLoader(train_loader=train_loader, val_loader=val_loader)
    
    posterior_ensemble, summaries = runner(loader=loader,)
    
    return posterior_ensemble, summaries


    
                       
    

    

    
    
    
    