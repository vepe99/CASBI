import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
import numpy as np
import pandas as pd
import torch
import os


import ili
from ili.dataloaders import  TorchLoader
from ili.inference import InferenceRunner
from ili.validation.metrics import PosteriorSamples
import tarp

from CASBI.utils.CNN import ConvNet_subhalo
from CASBI.inference import CustomDataset_subhalo
import CASBI.create_template_library as ctl

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
    hidden_features = trial.suggest_categorical('hidden_features', [50, 70, 80, 100, ])
    num_transforms = trial.suggest_categorical('num_transforms', [20, 25, 30,])
    output_dim = trial.suggest_categorical('output_dim', [25, 30, 35])
        
    prior = ili.utils.Uniform(low=[3.5, -2.], high=[10.6, 1.15], device=device)
    embedding_net = ConvNet_subhalo(output_dim=output_dim).to(device)
    
    nets = [
        ili.utils.load_nde_lampe(model=model, hidden_features=hidden_features, num_transforms=num_transforms,
                            embedding_net=embedding_net, x_normalize=False, device=device), ]
    train_args = {
    'training_batch_size': 256,
    'learning_rate': 1e-4,
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
    
    train_loader = torch.utils.data.DataLoader(CustomDataset_subhalo(train_data, train_targets,), shuffle=True, batch_size=256)
    val_loader = torch.utils.data.DataLoader(CustomDataset_subhalo(val_data, val_targets,), shuffle=False, batch_size=256)
    # test_loader = DataLoader(test_dataset,  shuffle=False)

    loader = TorchLoader(train_loader=train_loader, val_loader=val_loader) 

    # train a model to infer x -> theta. save it as toy/posterior.pkl
    # runner = InferenceRunner.from_config(f"./training.yaml")
    posterior, summaries = runner(loader=loader)
    
    
    try:
        sampler = PosteriorSamples(num_samples=2000, sample_method='direct')
    except Exception:
        sampler = PosteriorSamples(num_samples=2001, sample_method='direct')
        
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
    
    gpu_index = 0  # replace with your desired GPU index
    torch.cuda.set_device(gpu_index)
    device = f"cuda:{gpu_index}"
    
    #path to the files generetated by the CASBI.preprocessing
    data_path = "/export/data/vgiusepp/casbi_rewriting"

    galaxy_file_path = os.path.join(data_path, "new_files/")
    dataframe_path = os.path.join(data_path, "dataframe.parquet")
    preprocessing_path = os.path.join(data_path, "preprocess_file.npz")

    #generate template library
    sigma = 0.

    template_library = ctl.TemplateLibrary(galaxy_file_path=galaxy_file_path, 
                                           dataframe_path=dataframe_path, 
                                           preprocessing_path=preprocessing_path, 
                                           M_tot=1e10,
                                           sigma=sigma,)    
    template_library.create_single_halo_library(test_percentage=0.1)
    
    #get the training and test data
    x_train, params_train, x_test, params_test = template_library.get_inference_input()
    
    #convert to x and theta for the objective function
    x = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)  # Shape: (batch, 1, 64, 64)
    theta = params_train
    
    
    study_name = 'example_study'  # Unique identifier of the study.
    storage_name = 'sqlite:///example_onlylog.db'
    study = optuna.create_study(study_name=study_name, storage=storage_name,directions=['maximize', 'minimize'], load_if_exists=True)
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    study.optimize(objective, callbacks=[MaxTrialsCallback(110, states=(TrialState.COMPLETE,))],)
    