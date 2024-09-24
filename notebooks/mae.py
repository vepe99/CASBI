import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle
from ili.validation.metrics import PosteriorCoverage
from os.path import join
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor

def process_item(item):
    (i, j), value = item
    x = value['x']
    # Create additional channels filled with i and j
    i_channel = np.full_like(x, i)
    j_channel = np.full_like(x, j)
    # Concatenate the original x with the new channels
    x_with_channels = np.stack((x, i_channel, j_channel))
    
    theta = value['params']
    concatenated_theta = np.concatenate((theta, [i, j]))
    
    return x_with_channels, concatenated_theta, i, j

def process_dictionary(samples, sigma, data_dict, num_samples):
    

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_item, data_dict.items()))

    for x_with_channels, concatenated_theta, i, j in results:
        sample_i_j = {}
        sampled_theta = posterior_dict[m].sample((num_samples, ), x=torch.tensor(x_with_channels))[:, 0] 
        sample_i_j[(i, j)] = sampled_theta
        if sigma in samples.keys():
            samples[sigma].update(sample_i_j)
        else:
            samples[sigma] = sample_i_j

def run_with_retry(samples,sigma, data_dict, num_samples):
    while True:
        try:
            process_dictionary(samples, sigma, data_dict, num_samples)
            num_samples = 1001
            break  # Exit the loop if no exception is raised 
        except Exception as e:
            print(f"Error encountered: {e}. Retrying with num_samples={num_samples + 1}")
            num_samples += 1



samples = {}
for sigma in tqdm([0.0, 0.01, 0.02, 0.04, 0.5, 0.8]):
    with open(f'./script/uncertanties_testing/test_set/test_set_{sigma}.pkl', 'rb') as pickle_file:
        data_dict = pickle.load(pickle_file)
    posterior_dict = {}
    m = 'NPE'
    with open(join(f'./script/uncertanties_testing/posterior/posterior_{sigma}', 'posterior.pkl'), 'rb') as f:
        posterior_dict[m] = pickle.load(f)
        
    # Assuming posterior_dict['NPE'] is the object containing the given dictionary structure
    npe_object = posterior_dict['NPE']

    def set_device(module, device):
        """
        Recursively set the device for all submodules that have a 'to' method.
        """
        if hasattr(module, 'to'):
            module.to(device)
        for child in module.children():
            set_device(child, device)

    # Access the 'posteriors' ModuleList
    posteriors = npe_object._modules['posteriors']

    # Set the device for each LampeNPE object and its submodules
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    for i, lampe_npe in enumerate(posteriors):
        set_device(lampe_npe, device)
        embedding_net = lampe_npe.embedding_net
        print(f"Embedding Net for LampeNPE {i} is now on device: {device}")

    # Optionally, set the device for the main NPE object itself
    set_device(npe_object, device)
    num_samples = 1001
    run_with_retry(samples, sigma, data_dict, num_samples)
    
# Save the dictionary to a pickle file
with open('sample_sigma.pkl', 'wb') as file:
    pickle.dump(samples, file)

