import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
import yaml
import time
import numpy as np
import pandas as pd
import torch
import multiprocessing
from multiprocessing import Pool

from ili.dataloaders import StaticNumpyLoader, NumpyLoader
from ili.inference import InferenceRunner
from ili.validation import ValidationRunner
import ili

from CASBI.utils.create_dataframe import rescale
from CASBI.sbi.conf_yaml import generate_data_yaml, generate_training_yaml
from CASBI.sbi.gen_halo import  gen_halo_Nsubhalos, gen_halo
from CASBI.sbi.inference import run_inference, load_posterior, infer_observation, evaluate_posterior, calibrarion

import torch.nn as nn

device='cuda'

prior = ili.utils.Uniform(low=[2.0], high=[100.0], device=device)

class ConvNet(nn.Module):
    def __init__(self, input_channel, output_dim, extend_convolutions):
        super(ConvNet, self).__init__()
        if extend_convolutions:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(input_channel, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.fc_layers = nn.Sequential(
                nn.Linear(32 * 8 * 8, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim),
            )
        else:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(input_channel, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.fc_layers = nn.Sequential(
                nn.Linear(64 * 4 * 4, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim),
            )
            

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return out
    
    
def objective(trial):
    # Suggest values for the hyperparameters
    model = trial.suggest_categorical('model', ['nsf'])
    extend_convolutions = trial.suggest_categorical('extend_convolutions', [True, False])
    hidden_features = trial.suggest_categorical('hidden_features', [50, 70, 100, 150, 200])
    num_transforms = trial.suggest_categorical('num_transforms', [10, 15, 20, 30])
    learning_rate = trial.suggest_categorical('learning_rate', [1e-4, 1e-5]) #suggest_loguniform('learning_rate', 1e-5, 1e-4)
    output_dim = trial.suggest_categorical('output_dim', [64, 128, 256, 512])
        
    # reload all simulator examples as a dataloader
    # all_loader = StaticNumpyLoader.from_config("../../../../../../data/vgiusepp/complete_inference/N_subhalos_data.yaml")
    data_path = "/export/data/vgiusepp/optuna_N_subhalos/"
    all_loader = NumpyLoader(x=np.load(data_path + "x.npy"), theta=np.load(data_path + "N_subhalos.npy"))
    
    embedding_net = ConvNet(input_channel=1, output_dim=output_dim, extend_convolutions=extend_convolutions)
    
    nets = [ ili.utils.load_nde_sbi(engine='NPE', model=model, hidden_features=hidden_features, num_transforms=num_transforms, embedding_net=embedding_net)]

    train_args = {
    'training_batch_size': 1024,
    'learning_rate': learning_rate,
    'stop_after_epochs': 20}
    
    runner = InferenceRunner.load(
    backend='sbi',
    engine='NPE',
    prior=prior,
    nets=nets,
    device=device,
    embedding_net=embedding_net,
    train_args=train_args,
    proposal=None,
    out_dir=None,)

    # train a model to infer x -> theta. save it as toy/posterior.pkl
    # runner = InferenceRunner.from_config(f"./training.yaml")
    _, summaries = runner(loader=all_loader)
    
    return summaries[0]['validation_log_probs'][-1]


if __name__ == '__main__':
    # loading and rescaling the data
    data = pd.read_parquet('~/data/dataframe/dataframe.parquet')
    data = rescale(data, mean_and_std_path='~/data/preprocess/mean_and_std.parquet', scale_observations=True, scale_parameters=True, inverse=True) 
    data =  data.drop(['gas_log10mass', 'a','redshift', 'mean_metallicity', 'std_metallicity','mean_FeMassFrac', 'std_FeMassFrac', 'mean_OMassFrac', 'std_OMassFrac'], axis=1)
    data_path = "/export/data/vgiusepp/optuna_N_subhalos"
    _ = gen_halo_Nsubhalos(data=data,
                       output_dir=data_path,
                       n_test=100,
                       n_train=1000,)
    study_name = 'example_study'  # Unique identifier of the study.
    storage_name = 'sqlite:///example.db'
    study = optuna.create_study(study_name=study_name, storage=storage_name,direction='maximize', load_if_exists=True)
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    study.optimize(objective, callbacks=[MaxTrialsCallback(100, states=(TrialState.COMPLETE,))],)
    