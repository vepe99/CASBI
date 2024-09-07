"""
Wrapper around the ltu-ili packages for the inference task. For now only the StaticNumpyLoader is supported.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from ili.dataloaders import StaticNumpyLoader, TorchLoader, NumpyLoader
from ili.dataloaders import TorchLoader
from ili.inference import InferenceRunner
from ili.validation.metrics import PosteriorCoverage, PlotSinglePosterior

import pickle
import os

def run_inference(train_yaml_path:str, data_yaml_path:str):
    """
    Run the inference task using the given yaml files.
    
    Args:
    train_yaml_path (str): Path to the yaml file containing the training configuration.
    data_yaml_path (str): Path to the yaml file containing the data configuration.
    
    """
    all_loader = StaticNumpyLoader.from_config(data_yaml_path)
    runner = InferenceRunner.from_config(train_yaml_path)
    runner(loader=all_loader)
    
def load_posterior(filepath):
    """
    Load the inference runner from a pkl file.
    
    Args:
    filepath (str): Path to the pkl file containing the already trained inference runner .
    
    Returns:
    InferenceRunner: The already trained inference runner.
    """
    samples = {}
    m = 'NPE'
    with open(filepath, 'rb') as f:
        samples[m] = pickle.load(f)
    return samples[m]


def infer_observation(posterior, observation_path:str, n_samples:int=10_000):
    """
    Generate samples from the posterior distribution given an observation.
    
    Args:
    posterior (InferenceRunner): The trained inference runner.
    observation_path (str): Path to the observation file.
    n_samples (int): Number of samples to generate.
    
    Returns:
    torch.Tensor: The generated samples.
    """
    return posterior.sample((n_samples,), x=torch.from_numpy(np.load(observation_path)).to('cuda'))

def evaluate_posterior(posterior, observation_path:str, parameter_path:str, labels:list, n_samples:int=10_000, ):
    """
    Evaluate the posterior distribution given an observation and compare it the true parameter.
    
    Args:
    posterior (InferenceRunner): The trained inference runner.
    observation_path (str): Path to the observation file.
    parameter_path (str): Path to the true parameter file.
    n_samples (int): Number of samples to generate.
    labels (list): List of labels of the parameters.
    
    Returns:
    plt.figure: The plot of the posterior distribution and the true parameter. 
    """
    metric = PlotSinglePosterior(
        num_samples=n_samples, sample_method='direct', 
        labels = labels,
        out_dir=None)
    fig = metric(
        posterior=posterior,
        x_obs=torch.from_numpy(np.load(observation_path)), theta_fid=torch.from_numpy(np.load(parameter_path)))
    
    return fig

def calibrarion(posterior, observation_test_path:str, parameter_test_path:str, labels:list, n_samples:int=10_000, ):
    """
    Evaluate the posterior distribution given the test observation and compare it the true test parameters for calibration.
    
    Args:
    posterior (InferenceRunner): The trained inference runner.
    observation_test_path (str): Path to the test observation file.
    parameter_test_path (str): Path to the test parameter file.
    n_samples (int): Number of samples to generate.
    labels (list): List of labels of the parameters.
    
    Returns:
    list: The list of calibration plot of the posterior distribution and the true parameter. 
    
    """
    metric = PosteriorCoverage(
        num_samples=n_samples, sample_method='direct',
        labels=labels, plot_list = ["coverage", "histogram", "predictions", "tarp"])
    fig = metric(
        posterior=posterior,
        x=torch.from_numpy(np.load(observation_test_path)), theta=torch.from_numpy(np.load(parameter_test_path))
    )
    return fig
