import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
import yaml
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool
from multiprocessing import cpu_count

import ili
from ili.dataloaders import NumpyLoader, TorchLoader
from ili.inference import InferenceRunner
from ili.validation.metrics import PosteriorCoverage, PlotSinglePosterior
from ili.validation.metrics import PosteriorSamples
import tarp
from CNN import ConvNet

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn


from sklearn.neighbors import NearestNeighbors



    
def objective(trial):
    
    data_size = len(x)
    print(data_size)    
    indices = np.random.permutation(data_size)

    # Decide on the split size, for example 80% for training and 20% for validation
    split_idx = int(data_size * 0.8)

    # Split the indices
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]

    # Create the data splits
    train_data, train_targets = x[train_indices].float(), theta[train_indices].float()
    val_data, val_targets = x[val_indices].float(), theta[val_indices].float(),
   
    # Suggest values for the hyperparameters
    model = trial.suggest_categorical('model', ['nsf', 'gf', 'maf'])
    hidden_features = trial.suggest_categorical('hidden_features', [10, 50, 70, 100])
    num_transforms = trial.suggest_categorical('num_transforms', [10, 15, 20])
    output_dim = trial.suggest_categorical('output_dim', [10, 32, 64])
        
    prior = ili.utils.Uniform(low=minimum_theta, high=maximum_theta, device=device)
    embedding_net = ConvNet(output_dim=output_dim).to(device)
    
    nets = [
        ili.utils.load_nde_lampe(model=model, hidden_features=hidden_features, num_transforms=num_transforms,
                            embedding_net=embedding_net, x_normalize=False, device=device), ]
    train_args = {
    'training_batch_size': 2024,
    'learning_rate': 5e-5,
    'stop_after_epochs': 20}
    
    runner = InferenceRunner.load(
    backend='lampe',
    engine='NPE',
    prior=prior,
    nets=nets,
    device=device,
    train_args=train_args,
    proposal=None,
    out_dir=None,
    )
    
    train_loader = torch.utils.data.DataLoader(CustomDataset(train_data, train_targets,), shuffle=True, batch_size=2024)
    val_loader = torch.utils.data.DataLoader(CustomDataset(val_data, val_targets,), shuffle=False, batch_size=2024)
    # test_loader = DataLoader(test_dataset,  shuffle=False)

    loader = TorchLoader(train_loader=train_loader, val_loader=val_loader) 

    # train a model to infer x -> theta. save it as toy/posterior.pkl
    # runner = InferenceRunner.from_config(f"./training.yaml")
    posterior, summaries = runner(loader=loader)
    
    
    sampler = PosteriorSamples(num_samples=500, sample_method='direct')
    xv, tv = val_data.to('cuda'), val_targets.to('cuda')
    samps = sampler(posterior, xv, tv)
    
    # measure tarp
    ecp, alpha = tarp.get_tarp_coverage(
        samps, tv.cpu().numpy(),
        norm=True, bootstrap=True,
        num_bootstrap=100
    )

    tarp_val = torch.mean(torch.from_numpy(ecp[:,ecp.shape[1]//2])).to('cuda')
    
    return summaries[0]['validation_log_probs'][-1], abs(tarp_val-0.5)


    
def pdf(m, alpha):
        norm_const = (m_max**(1-alpha) - m_min**(1-alpha))/(1-alpha) 
        return (1/norm_const)* m**(-alpha)
def cdf(m, alpha):
    norm_const = (m_max**(1-alpha) - m_min**(1-alpha))/(1-alpha) 
    return (1/norm_const)* (1/(1-alpha)) * (m**(1-alpha) - m_min**(1-alpha))

def inverse_cdf(y, alpha):
    norm_const = (m_max**(1-alpha) - m_min**(1-alpha))/(1-alpha) 
    return (y*norm_const*(1-alpha) + m_min**(1-alpha))**(1/(1-alpha))

def gen_non_repeated_halo(samples, masses, times, M_tot, nbrs, d):
    iteration = 0
    while iteration < 100: #number of max halos to be sampled
        if M_tot < mass_nn.min():
            break
        max_u = cdf(M_tot, alpha)
        analictical_sample = inverse_cdf(np.random.uniform(0, max_u), alpha, ).reshape(-1, 1)
        distances, indices = nbrs.kneighbors(analictical_sample)
        sample = galaxy_name[indices[0]][0][0]
        mass_sample = mass_nn[indices[0]][0][0]
        time_sample = infall_time[indices[0]][0][0]
        if (abs(mass_sample - analictical_sample) > d*analictical_sample) | (sample in samples):
            nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(mass_nn)
            analytic_10_samples = inverse_cdf(np.random.uniform(0, 1, size=5), alpha, ).reshape(-1, 1)
            distances, indices = nbrs.kneighbors(analytic_10_samples)
            galaxy_10 = galaxy_name[indices]
            mass_10 = mass_nn[indices]
            time_10 = infall_time[indices]
            mask =  (distances < d*analytic_10_samples).reshape(galaxy_10.shape)&(~np.isin(galaxy_10, samples))
            if (mask.sum() == 0):
                if (((1.4*1e9-M_tot)/(1.4*1e9))<0.95):
                    # print(f'No halos satified the requirement, total mass is: {((6*1e9-M_tot)/(6*1e9))*100:.2f} %') #this is for studying rejection of not completed galaxy
                    samples = None
                    masses = None
                    times = None
                    return samples, masses, times
                else:
                    break #when the 95% of the total mass is reach we keep only those samples and we do not add more
            else:
                sampled_index = np.random.choice(range(len(mass_10[mask].flatten())))
                mass_sample =  mass_10[mask].flatten()[sampled_index]
                sample = galaxy_10[mask].flatten()[sampled_index]
                time_sample  = time_10[mask].flatten()[sampled_index]
        samples.append(sample)
        masses.append(mass_sample)
        times.append(time_sample)

        M_tot = M_tot - mass_sample
        iteration += 1 
    return samples, masses, times


def gen_real_halo(j, galaxy_name, mass_nn, infall_time, galaxies_test=None, d=0.1, ):
    np.random.seed(j)
    N=2
    nbrs = NearestNeighbors(n_neighbors=N, algorithm='ball_tree').fit(mass_nn)
    M_tot = 1.4 * 1e9
    samples = []
    masses =  []
    times = []
    #generate a milky way galaxy like halo
    samples, masses, times = gen_non_repeated_halo(samples, masses, times, M_tot, nbrs, d)
    
    #check if the milky way like halo is in the test set, otherwise genereate a new one untill is not present anymore in the test set
    if (galaxies_test is not None)&(samples is not None):
        while any(set(samples) == galaxy_in_testset for galaxy_in_testset in galaxies_test):
            samples = []
            masses =  []
            times = []
            samples, masses, times = gen_non_repeated_halo(samples, masses, times, M_tot, nbrs, d)
    if samples is None:
        return np.array([]), np.array([]), np.array([])

    #get the galaxy name to load the histogram from memory 
    samples =  np.array(samples)
    arr = np.array([np.load('/export/data/vgiusepp/data/full_dataframe/histogram_data/'+f'{s}'+'.npz' )['observables']  for s in samples ])
    #some all the histogram to obtain the 0th channel 
    hist_0 = np.sum( arr, axis=0)
    hist_to_return = [np.stack([hist_0, np.ones_like(hist_0)*i]) for i in range(samples.shape[0])]  #nasty trick to allow to save both the N_th number and the histogram in the same array
    
    masses = np.array(masses)
    infall_time = np.array(infall_time)
    indices = np.argsort(masses)[::-1] #sort the masses in descending order
    
    #reorder masses and infall time
    masses = masses[indices]
    infall_time = infall_time[indices]
    samples = samples[indices]
    
    return hist_to_return, np.column_stack([masses, infall_time]), np.array([samples for i in range(samples.shape[0])]) # I want for each of the hist to have all the names of the galaxies that contributed to it, I cannot flatten it 

class CustomDataset(Dataset):
    def __init__(self, observation, parameters, ):
        self.observation = observation
        self.parameters = parameters
        
        self.tensors = [self.observation, self.parameters]

    def __len__(self):
        return len(self.observation)

    def __getitem__(self, idx):
        observation = self.observation[idx].to('cuda') #this should put just the batch on the gpu
        parameters = self.parameters[idx].to('cuda')
        
        # observation = self.observation[idx] #this should put just the batch on the gpu
        # parameters = self.parameters[idx]

        return observation, parameters
        
if __name__ == '__main__':
    
    gpu_index = 1  # replace with your desired GPU index
    torch.cuda.set_device(gpu_index)
    device = f"cuda:{gpu_index}"

    data = pd.read_parquet('/export/data/vgiusepp/data/full_dataframe/dataframe/dataframe.parquet')
    data['star_log10mass'] = 10**data['star_log10mass']
    data_mass = data['star_log10mass'].drop_duplicates()
    data_mass = data_mass[data_mass<1.4*1e9]
    
    m_min, m_max = data_mass.min(), data_mass.max()
    alpha = 1.25
   
    mass_name = data[['star_log10mass', 'Galaxy_name', 'infall_time']].drop_duplicates()
    mass_name = mass_name[mass_name['star_log10mass']<1.4*1e9]
    min_feh, max_feh = data['feh'].min(), data['feh'].max() 
    min_ofe, max_ofe = data['ofe'].min(), data['ofe'].max()
    mass_nn = mass_name['star_log10mass'].values.reshape(-1, 1)
    infall_time = mass_name['infall_time'].values.reshape(-1, 1)
    galaxy_name = mass_name['Galaxy_name'].values.reshape(-1, 1)
    
    test_set_sample = 100
    train_set_sample = 5_000

    mass_nn = mass_name['star_log10mass'].values.reshape(-1, 1)
    infall_time = mass_name['infall_time'].values.reshape(-1, 1)
    galaxy_name = mass_name['Galaxy_name'].values.reshape(-1, 1)
    with Pool(processes=cpu_count()) as p:
        result = p.starmap(gen_real_halo, [[j, galaxy_name, mass_nn, infall_time] for j in range(test_set_sample)]   )

    hist_list_test, params_list_test, galaxy_list_test = zip(*result)

    #create the filter to take only the unique galaxies 
    single_galaxy_test = [arr[0] for arr in galaxy_list_test if arr.size > 0]
    unique_indices_test = list({tuple(arr): i for i, arr in enumerate(map(tuple, single_galaxy_test))}.values())
    print('unique galaxy in the test set that are not empty:', len(unique_indices_test))

    flattened_hist_list_test = [item for i, sublist in enumerate(hist_list_test) if i in unique_indices_test for item in sublist]
    flattened_param_list_test = [item for i, sublist in enumerate(params_list_test) if i in unique_indices_test for item in sublist]
    flattened_hist_list_test =  np.array(flattened_hist_list_test)
    flattened_param_list_test =  np.array(flattened_param_list_test)
    galaxies_test = [set(arr[0]) for arr in galaxy_list_test if arr.size > 0] #list that contains set of names of the galaxy in the test set to compare it with the training set 


    with Pool(processes=cpu_count()) as p:
        result = p.starmap(gen_real_halo, [[j+test_set_sample, galaxy_name, mass_nn, infall_time, galaxies_test] for j in range(train_set_sample)]   )

    hist_list, params_list, galaxy_list = zip(*result)

    #create the filter to take only the unique galaxies 
    single_galaxy = [arr[0] for arr in galaxy_list if arr.size > 0]
    unique_indices = list({tuple(arr): i for i, arr in enumerate(map(tuple, single_galaxy))}.values())
    flattened_hist_list = [item for i, sublist in enumerate(hist_list) if i in unique_indices for item in sublist]
    flattened_param_list = [item for i, sublist in enumerate(params_list) if i in unique_indices for item in sublist]
    flattened_hist_list =  np.array(flattened_hist_list)
    flattened_param_list =  np.array(flattened_param_list)
    print('unique galaxy in the training set that are not empty:', len(unique_indices))
    
    mask = [flattened_hist_list[:, 1, 0, 0] < np.random.uniform(low=0, high=100, size=len(flattened_hist_list[:, 1, 0, 0])) ][0] #applying a mask to not overfit on high N
    # mask = [flattened_hist_list[:, 1, 0, 0] < 20 ][0] #applying a mask to not overfit on high N
    training_x = flattened_hist_list[mask]
    training_theta = flattened_param_list[mask]

    test_x = torch.from_numpy(flattened_hist_list_test)
    # test_x[:, 0, :, :] = test_x[:, 0, :, :]/test_x[:, 0, :, :].sum()
    test_x = torch.log1p(test_x).float()
    test_theta = torch.log10(torch.from_numpy(flattened_param_list_test)).float()

    x = torch.from_numpy(training_x)
    # x[:, 0, :, :] = x[:, 0, :, :]/x[:, 0, :, :].sum()
    x = torch.log1p(torch.from_numpy(training_x)).float()
    theta = torch.log10(torch.from_numpy(training_theta)).float()
    

    conditions  = mass_name[['star_log10mass', 'infall_time']]
    minimum_theta = [np.log10(conditions[col].values.min()) for col in conditions.columns]   
    maximum_theta = [np.log10(conditions[col].values.max()) for col in conditions.columns]    

    
    def write_to_yaml(minimum_theta, maximum_theta, device):
        # Load the existing data
        with open('./training.yaml', 'r') as file:
            data = yaml.safe_load(file)
            
        # Update the value
        # data['prior']['args']['low'] = minimum_theta[0]
        # data['prior']['args']['high'] = maximum_theta[0]
        data['device'] = device

        # Write the data back to the file
        with open('./training.yaml', 'w') as file:
            yaml.safe_dump(data, file)
            
    # write_to_yaml(minimum_theta, maximum_theta, device)
    print('write the right prior in the training.yaml file')        

    
    
    study_name = 'example_study'  # Unique identifier of the study.
    storage_name = 'sqlite:///example_onlylog.db'
    study = optuna.create_study(study_name=study_name, storage=storage_name,directions=['maximize', 'minimize'], load_if_exists=True)
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    study.optimize(objective, callbacks=[MaxTrialsCallback(200, states=(TrialState.COMPLETE,))],)
    