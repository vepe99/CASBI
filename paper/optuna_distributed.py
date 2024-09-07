import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
import numpy as np
import pandas as pd
import torch


import ili
from ili.dataloaders import  TorchLoader
from ili.inference import InferenceRunner
from ili.validation.metrics import PosteriorSamples
import tarp
from CNN import ConvNet

from CASBI.sbi.inference import CustomDataset
from CASBI.utils import create_template_libary as ctl

##OPTUNA DISTRIBUTED MULTI OBJECTIVE OPTIMIZATION STUDY

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
        
    prior = ili.utils.Uniform(low=[3.5, -2.], high=[10, 1.15], device=device)
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


if __name__ == '__main__':
    
    gpu_index = 1  # replace with your desired GPU index
    torch.cuda.set_device(gpu_index)
    device = f"cuda:{gpu_index}"
    
    data = pd.read_parquet('/export/data/vgiusepp/data/full_dataframe/dataframe/dataframe.parquet')
    M_tot = 1.4*1e9
    alpha = 1.25
    test_set_sample = 100
    training_set_sample = 1_000


    #generate the input for the creation of the training and test template library 
    galaxy_name, mass_nn, infall_time, m_max, m_min = ctl.template_input(data, M_tot=1.4*1e9)

        
    #generate the test library
    flattened_hist_list_test, flattened_param_list_test, galaxies_test_name = ctl.gen_template_library(test_set_sample, galaxy_name, M_tot, mass_nn, infall_time, m_max, m_min, alpha)

    #generate the training library
    flattened_hist_list, flattened_param_list, galaxies_train_name = ctl.gen_template_library(training_set_sample, galaxy_name, M_tot, mass_nn, infall_time, m_max, m_min, alpha, galaxies_test_name)

    #rebalance the training set to avoid overfitting on high N
    mask = [flattened_hist_list[:, 1, 0, 0] < np.random.uniform(low=0, high=100, size=len(flattened_hist_list[:, 1, 0, 0])) ][0] 
    training_x = flattened_hist_list[mask]
    training_theta = flattened_param_list[mask]

        
    #test set for 
    test_x = torch.from_numpy(flattened_hist_list_test).float()
    test_theta = torch.from_numpy(flattened_param_list_test).float()

    #training set 
    x = torch.from_numpy(training_x).float()
    theta = torch.from_numpy(training_theta).float()
    
    study_name = 'example_study'  # Unique identifier of the study.
    storage_name = 'sqlite:///example_onlylog.db'
    study = optuna.create_study(study_name=study_name, storage=storage_name,directions=['maximize', 'minimize'], load_if_exists=True)
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    study.optimize(objective, callbacks=[MaxTrialsCallback(200, states=(TrialState.COMPLETE,))],)
    