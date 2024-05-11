import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import time
import os
import re
import itertools
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
         
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split 
import optuna

from  CASBI.generator.nf.nf_model import *
        
def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

class Trainer:
    '''
    Class for training a model with distributed data parallelism.
    
    Args:
        model (NF_condGLOW): The model to be trained.
        train_data (torch.utils.data.DataLoader): The data loader for training data.
        val_data (torch.utils.data.DataLoader): The data loader for validation data.
        test_data (torch.utils.data.DataLoader): The data loader for test data.
        optimizer (torch.optim.Optimizer): The optimizer for model parameters.
        snapshot_path (str): The path to save the training snapshots.
        
    Attributes:
        gpu_id (int): The ID of the GPU being used.
        model (NF_condGLOW): The model to be trained.
        train_data (torch.utils.data.DataLoader): The data loader for training data.
        val_data (torch.utils.data.DataLoader): The data loader for validation data.
        test_data (torch.utils.data.DataLoader): The data loader for test data.
        best_loss (float): The best validation loss achieved during training.
        optimizer (torch.optim.Optimizer): The optimizer for model parameters.
        epochs_run (int): The number of epochs already run.
        snapshot_path (str): The path to save the training snapshots.
        logger (SummaryWriter): The logger for training progress.
    
    Methods:
        _load_snapshot(snapshot_path): Loads a training snapshot from a given path.
        _run_batch(source, train=True): Runs a batch of data through the model and computes the loss.
        _run_epoch(epoch): Runs an epoch of training and validation.
        _save_checkpoint(epoch): Saves a training snapshot at a given epoch.
        train(max_epochs): Trains the model for a maximum number of epochs.
        test(test_set): Evaluates the model on a test set.
    '''
    def __init__(
        self,
        model,
        train_data: torch.utils.data.DataLoader,
        val_data: torch.utils.data.DataLoader,
        test_data: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,  
        snapshot_path: str,) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.best_loss = 1_000
        self.optimizer = optimizer
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)
            
        if self.gpu_id == 0:
            self.logger = SummaryWriter()
        else:
            self.logger = None
            
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    
    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        if os.environ["RANK"] == 0:
            print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, train=True):
        if train==True:
            mask_cond = np.ones(source.shape[1]).astype(bool)
            mask_cond[:2] = np.array([0, 0]).astype(bool)
            mask_cond = torch.from_numpy(mask_cond).to(self.gpu_id)
            #Evaluate model
            z, logdet, prior_z_logprob = self.model(source[..., ~mask_cond], source[..., mask_cond])
            
            #Get loss
            loss = -torch.mean(logdet+prior_z_logprob) 
            #Set gradients to zero
            self.optimizer.zero_grad()
            #Compute gradients
            loss.backward()
            #Update parameters
            self.optimizer.step()
            return loss.item()
        else:
            mask_cond = np.ones(source.shape[1]).astype(bool)
            mask_cond[:2] = np.array([0, 0]).astype(bool)
            mask_cond = torch.from_numpy(mask_cond).to(self.gpu_id)
            #Evaluate model
            z, logdet, prior_z_logprob = self.model(source[..., ~mask_cond], source[..., mask_cond])
            
            #Get loss
            loss = -torch.mean(logdet+prior_z_logprob)
            return loss.item()

    def _run_epoch(self, epoch):
        b_sz = self.train_data.batch_size
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch) # shuffle data
        train_loss = 0
        for source in self.train_data:
            source = source.to(self.gpu_id)
            train_loss += self._run_batch(source, train=True)/len(self.train_data)     
        train_loss = torch.tensor([train_loss]).to(self.gpu_id)
        
        self.val_data.sampler.set_epoch(epoch)    
        self.model.eval()
        val_running_loss = 0.
        with torch.no_grad():
            for source in self.val_data:
                source = source.to(self.gpu_id)
                batch_loss = self._run_batch(source, train=False)
                val_running_loss += batch_loss/len(self.val_data)
        val_running_loss = torch.tensor([val_running_loss]).to(self.gpu_id)
        
        dist.barrier()
        dist.all_reduce(val_running_loss, op=dist.ReduceOp.SUM)    
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        if self.gpu_id == 0:
            self.logger.add_scalar("Loss/val", val_running_loss/int(os.environ["WORLD_SIZE"]), epoch) #the WORLD_SIZE is the number of GPUs
            self.logger.add_scalar("Loss/train", train_loss/int(os.environ["WORLD_SIZE"]), epoch) #the WORLD_SIZE is the number of GPUs
        dist.barrier()
        
        self.model.train()
        return val_running_loss/int(os.environ["WORLD_SIZE"])
            
    def _save_checkpoint(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            val_running_loss = self._run_epoch(epoch)
            if val_running_loss < self.best_loss and self.gpu_id == 0 :  
                self.best_loss = val_running_loss
                self._save_checkpoint(epoch)
            

def get_even_space_sample(df_mass_masked):
    '''
    Given a dataframe of galaxy in a range of mass, it returns 10 equally infall time spaced samples  
    '''
    len_infall_time = len(df_mass_masked['infall_time'].unique())
    index_val_time = np.linspace(0, len_infall_time-1, 10)
    time = np.sort(df_mass_masked['infall_time'].unique())[index_val_time.astype(int)]
    for i, t in enumerate(time):
        temp = df_mass_masked[df_mass_masked['infall_time']==t]
        galaxy_temp = temp.sample(1)['Galaxy_name'].values[0]
        if i == 0:
            df_time = df_mass_masked[df_mass_masked['Galaxy_name']==galaxy_temp]
        else:  
            df_galaxy = df_mass_masked[df_mass_masked['Galaxy_name']==galaxy_temp]
            df_time = pd.concat([df_time, df_galaxy], ignore_index=True)
    return df_time
    
    
def load_train_objs(path_train_dataframe:str, test_and_nll_path:str):
    train_set = pd.read_parquet(path_train_dataframe)
    train_set = train_set[train_set.columns.difference(['a'], sort=False)] # load your dataset
    train_set = train_set[train_set.columns.difference(['redshift'], sort=False)]
    train_set = train_set[train_set.columns.difference(['mean_FeMassFrac'], sort=False)]
    train_set = train_set[train_set.columns.difference(['std_FeMassFrac'], sort=False)]
    train_set = train_set[train_set.columns.difference(['mean_OMassFrac'], sort=False)]
    train_set = train_set[train_set.columns.difference(['std_OMassFrac'], sort=False)]
    # Galax_name = train_set['Galaxy_name'].unique()
    # test_galaxy = np.random.choice(Galax_name, int(len(Galax_name)*0.1), replace=False)
    # test_set = train_set[train_set['Galaxy_name'].isin(test_galaxy)]
    # test_set.to_parquet('/export/home/vgiusepp/MW_MH/data/test_set.parquet')
    # train_set = train_set[~(train_set['Galaxy_name'].isin(test_galaxy))][train_set.columns.difference(['Galaxy_name'], sort=False)]
    # test_set = test_set[train_set.columns.difference(['Galaxy_name'], sort=False)]
    # test_set = torch.from_numpy(test_set.values)
    # train_set = torch.from_numpy(train_set.values)
    # train_set, val_set = train_test_split(train_set, test_size=0.2, random_state=42)
    
    low_percentile_mass, high_percentile_mass = np.percentile(train_set['star_log10mass'], 25), np.percentile(train_set['star_log10mass'], 75)
    low_mass = get_even_space_sample(train_set[train_set['star_log10mass']<=low_percentile_mass])
    intermediate_mass= get_even_space_sample(train_set[(train_set['star_log10mass']>low_percentile_mass) & (train_set['star_log10mass']<high_percentile_mass)])
    high_mass = get_even_space_sample(train_set[train_set['star_log10mass']>=high_percentile_mass])
    val_set = pd.concat([low_mass, intermediate_mass, high_mass])
    train_set = train_set[~train_set['Galaxy_name'].isin(val_set['Galaxy_name'])]
    
    low_percentile_mass, high_percentile_mass = np.percentile(train_set['star_log10mass'], 25), np.percentile(train_set['star_log10mass'], 75)
    low_mass = get_even_space_sample(train_set[train_set['star_log10mass']<=low_percentile_mass])
    intermediate_mass = get_even_space_sample(train_set[(train_set['star_log10mass']>low_percentile_mass) & (train_set['star_log10mass']<high_percentile_mass)])
    high_mass = get_even_space_sample(train_set[train_set['star_log10mass']>=high_percentile_mass])
    test_set = pd.concat([low_mass, intermediate_mass, high_mass])
    test_set.to_parquet(f'{test_and_nll_path}test_set.parquet')
    
    train_set = train_set[~train_set['Galaxy_name'].isin(test_set['Galaxy_name'])]
    #remove the column Galaxy name before passing it to the model
    test_set = test_set[train_set.columns.difference(['Galaxy_name'], sort=False)]
    train_set = train_set[train_set.columns.difference(['Galaxy_name'], sort=False)]
    val_set = val_set[train_set.columns.difference(['Galaxy_name'], sort=False)]
    test_set = torch.from_numpy(np.array(test_set.values, dtype=float))
    val_set = torch.from_numpy(np.array(val_set.values, dtype=float))
    train_set = torch.from_numpy(np.array(train_set.values, dtype=float))
    print('finish prepare data')
    conditions = train_set.shape[1] - 2
    model = NF_condGLOW(12, dim_notcond=2, dim_cond=conditions, CL=NSF_CL2, network_args=[256, 3, 0.2])  # load your model
    optimizer = torch.optim.RAdam(model.parameters(), lr=1e-4*int((os.environ["WORLD_SIZE"])))
    return train_set, val_set, test_set, model, optimizer     

def prepare_dataloader(dataset, batch_size: int):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset))

def main(path_train_dataframe: str, 
         test_and_nll_path: str,
         total_epochs: int, 
         batch_size: int,
         snapshot_path: str = "./snapshot/snapshot.pt",
         ):
    ddp_setup()
    train_set, val_set, test_set, model, optimizer = load_train_objs(path_train_dataframe, test_and_nll_path)
    train_data = prepare_dataloader(train_set, batch_size)
    val_data = prepare_dataloader(val_set, batch_size)
    test_data = prepare_dataloader(test_set, batch_size)
    trainer = Trainer(model, train_data, val_data, test_data, optimizer, snapshot_path)
    trainer.train(total_epochs)
    destroy_process_group()
    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('path_train_dataframe', type=str, help='Path to the training dataframe in parquet format')
    parser.add_argument('test_and_nll_path', type=str, help='Path to save the test set and the negative log likelihood')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('batch_size', default=256, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('snapshot_path', default="./snapshot/snapshot.pt", type=str, help='Path to save the training snapshots')
    args = parser.parse_args()


    begin=time.time()
    main(args.path_train_dataframe, args.test_and_nll_path, args.total_epochs, args.batch_size, args.snapshot_path,)
    end = time.time()
    print('total time', (end-begin)/60, 'minutes')