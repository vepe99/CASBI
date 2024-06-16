import os 
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
mpl.style.use('../paper.mcstyle')

import torch

from CASBI.utils.create_dataframe import rescale
from CASBI.sbi.conf_yaml import generate_data_yaml, generate_training_yaml
from CASBI.sbi.gen_halo import  gen_halo_Nsubhalos_hist, gen_halo_hist
from CASBI.sbi.inference import run_inference, load_posterior, infer_observation, evaluate_posterior, calibrarion


def create_subfolders_and_run(base_dir):
    """
    Create subfolders in the base directory and run a function within each subfolder.

    Parameters:
    base_dir (str): The base directory where subfolders will be created.
    num_subfolders (int): The number of subfolders to create.
    function_to_run (function): The function to run within each subfolder.
    *args: Additional arguments to pass to the function_to_run.
    """
    # for i in [2, 3, 5, 10, 20, 30]:
    for i in [2]:
        subfolder_path = os.path.join(base_dir, f'N_subhalos_{i}')
        os.makedirs(subfolder_path, exist_ok=True)
        
        # Change working directory to the subfolder
        os.chdir(subfolder_path)
        
        os.makedirs('N_subhalos_data', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
        #yaml file 
        
        #Number of subhalos inference
        generate_data_yaml(filepath='./N_subhalos_data.yaml', 
                        in_dir='./N_subhalos_data', #where the data are stored
                        theta_file='N_subhalos.npy',
                        thetafid_file='N_subhalos_0.npy')
        generate_training_yaml(filepath='./N_subhalos_training.yaml', output_file='./N_subhalos_NPE', hidden_feature=70, num_transforms=20, model='nsf', embedding_net='CNN', output_dim=128)


        #Subhalos properties inference
        generate_data_yaml(filepath='./data.yaml', in_dir='./data',) #where the data are stored
        generate_training_yaml(filepath='./training.yaml', output_file='./galaxy_NPE', hidden_feature=70, num_transforms=25, model='nsf', embedding_net='CNN', output_dim=128)
        
        # loading and rescaling the data
        data_dir = '../../data/full_dataframe/histogram_data/'
        
        #unseen galaxies to do infernce on when the whole pipeline is ready
        inference_N_subhalos = i
        inference_galaxy = set(random.sample(os.listdir(data_dir), inference_N_subhalos))
        inference_parameters =  np.array([[np.load(os.path.join(data_dir, g))['star_log10mass'], np.load(os.path.join(data_dir, g))['dm_log10mass'], np.load(os.path.join(data_dir, g))['infall_time']]  for g in inference_galaxy ]).T
        sorted_index = np.argsort(inference_parameters[0], )[::-1] #orders the parameters in descending order of star mass
        inference_parameters = (inference_parameters[:,sorted_index]).reshape(-1)
        # infererence_sim_data =  np.log10(np.sum(np.array([np.load(os.path.join(data_dir, g))['observables'] for g in inference_galaxy ]), axis=0)[np.newaxis, :]+1+1e-6)
        infererence_sim_data =  np.sum(np.array([np.load(os.path.join(data_dir, g))['observables'] for g in inference_galaxy ]), axis=0)[np.newaxis, :]


        np.save(os.path.join('./', 'inference_N_subhalos.npy'), np.array(inference_N_subhalos).reshape(1, 1))
        np.save(os.path.join('./', 'inference_theta.npy'), inference_parameters)
        np.save(os.path.join('./', 'inference_x.npy'), infererence_sim_data)
        np.save(os.path.join('./', 'inference_galaxy.npy'), inference_galaxy)
        
        #remove the galaxies
        # data = data[~data['Galaxy_name'].isin(np.load(os.path.join('./', 'inference_galaxy.npy'), allow_pickle=True))]
        
        _ = gen_halo_Nsubhalos_hist(data_dir=data_dir,
                       output_dir=os.path.join('./', 'N_subhalos_data' ),
                       inference_galaxy=inference_galaxy,
                       n_test=100,
                       n_train=1000,)
        
        # #train the posterior
        run_inference('./N_subhalos_training.yaml', './N_subhalos_data.yaml')
        
        with open('N_subhalos_NPE_summary.json', 'r') as f:
            # Load the data from the file
            summaries = json.load(f)
            
        fig, ax = plt.subplots(1, 1, figsize=(6,4))
        c = list(mcolors.TABLEAU_COLORS)
        for i, m in enumerate(summaries):
            ax.plot(m['training_log_probs'], ls='-', label=f"{i}_train", c=c[i])
            ax.plot(m['validation_log_probs'], ls='--', label=f"{i}_val", c=c[i])
        ax.set_xlim(0)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Log probability')
        ax.legend()
        fig.savefig('N_subhalos_training.png')

        #load the posterior from pkl file 
        posterior = load_posterior('./N_subhalos_NPE_posterior.pkl')
        
        fig = evaluate_posterior(posterior, observation_path=os.path.join('./', 'inference_x.npy'), parameter_path=os.path.join('./', 'inference_N_subhalos.npy'), labels=['N subhalos'], n_samples=10_000)
        fig.savefig('N_subhalos_evaluation.png')
        
        N_subhalos_samples = infer_observation(posterior, observation_path=os.path.join('./', 'inference_x.npy'), n_samples=10_000)
        np.save(os.path.join('./', 'N_subhalos_samples.npy'), N_subhalos_samples.cpu().numpy())
        
        fig = calibrarion(posterior=posterior, observation_test_path=os.path.join('./', 'N_subhalos_data/x_test.npy'), parameter_test_path=os.path.join('./', 'N_subhalos_data/N_subhalos_test.npy'), n_samples=10_000, labels=['N subhalos'])    
        fig[0].savefig('Calibration_0.png')
        fig[1].savefig('Calibration_1.png')
        fig[2].savefig('Calibration_2.png')
        fig[3].savefig('Calibration_3.png')
        
        full_dataset = pd.read_parquet('/export/data/vgiusepp/data/full_dataframe/dataframe/dataframe.parquet')
        prior_limits = pd.DataFrame({'star_log10mass': [full_dataset['star_log10mass'].min(), full_dataset['star_log10mass'].max()],
                                     'dm_log10mass': [full_dataset['dm_log10mass'].min(), full_dataset['dm_log10mass'].max()],
                                     'infall_time': [full_dataset['infall_time'].min(), full_dataset['infall_time'].max()]})
        
        # _ = gen_halo_hist(data_dir=data_dir, output_dir=os.path.join('./','data' ), 
        #      training_yaml='./training.yaml', #needs to be changed accordingly to how many N_subhalos were inferred
        #      inference_galaxy=inference_galaxy,
        #      n_test=1000, n_train=100_000, N_subhalos=round(N_subhalos_samples.mean().item()), prior_limits=prior_limits)
        _ = gen_halo_hist(data_dir=data_dir, output_dir=os.path.join('./','data' ), 
             training_yaml='./training.yaml', #needs to be changed accordingly to how many N_subhalos were inferred
             inference_galaxy=inference_galaxy,
             n_test=1000, n_train=500_000, N_subhalos=2, prior_limits=prior_limits)
        
        #train the posterior
        run_inference('./training.yaml', './data.yaml')
        
        # Open the JSON file
        with open('galaxy_NPE_summary.json', 'r') as f:
            # Load the data from the file
            summaries = json.load(f)
            
        fig, ax = plt.subplots(1, 1, figsize=(6,4))
        c = list(mcolors.TABLEAU_COLORS)
        for i, m in enumerate(summaries):
            ax.plot(m['training_log_probs'], ls='-', label=f"{i}_train", c=c[i])
            ax.plot(m['validation_log_probs'], ls='--', label=f"{i}_val", c=c[i])
        ax.set_xlim(0)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Log probability')
        ax.legend()
        fig.savefig('galaxy_training.png')

        #load the posterior from pkl file 
        posterior = load_posterior('./galaxy_NPE_posterior.pkl')
        
        
        inference_N_subhalos=np.load(os.path.join('./', 'inference_N_subhalos.npy'))
        # inference_parameters = np.load(os.path.join('./', 'inference_theta.npy'))
        # if inference_N_subhalos > round(N_subhalos_samples.mean().item()):
        #     inference_parameters = inference_parameters.reshape((3, inference_N_subhalos))
        #     inference_parameters = inference_parameters[:, :round(N_subhalos_samples.mean().item())].reshape(-1)
        # else:
        #     inference_parameters = inference_parameters.reshape((3, inference_N_subhalos.item()))
        #     inference_parameters = np.hstack((inference_parameters, np.zeros((3, round(N_subhalos_samples.mean().item()) - inference_N_subhalos.item())))).reshape(-1)

        # labels = np.array([[rf'$\log_{{10}}(M_{{s, {i}}})\ [M_\odot]$', rf'$\log_{{10}}(M_{{DM, {i}}})\ [M_\odot]$', rf'$\tau_{i}\ [Gyr]$'] for i in range(round(N_subhalos_samples.mean().item()))] )
        # labels = labels.T.reshape(-1)
        
        labels = np.array([[rf'$\log_{{10}}(M_{{s, {i}}})\ [M_\odot]$', rf'$\log_{{10}}(M_{{DM, {i}}})\ [M_\odot]$', rf'$\tau_{i}\ [Gyr]$'] for i in range(2)] )
        labels = labels.T.reshape(-1)

        np.save(os.path.join('./', 'inference_theta_Nsubahalosinfo.npy'), inference_parameters)

        #evaluate posterior
        fig = evaluate_posterior(posterior, 
                                observation_path=os.path.join('./', 'inference_x.npy'), parameter_path=os.path.join('./', 'inference_theta_Nsubahalosinfo.npy'),
                                labels=labels, n_samples=10_000)
        fig.savefig('./galaxy_evaluation.png')
        

if __name__ == "__main__":
    create_subfolders_and_run("/export/data/vgiusepp/analysis_hist/")