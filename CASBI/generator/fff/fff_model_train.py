import fff
from CASBI.generator.fff.fff_model import *
import torch

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ndtest

def get_even_space_sample(df_mass_masked):
    '''
    Given a dataframe of galaxy in a range of mass, it returns 10 equally infall time spaced samples  
    '''
    len_infall_time = len(df_mass_masked['infall_time'].unique())
    index_val_time = np.linspace(0, len_infall_time-1, 30)
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
    
    
def load_train_objs(path_train_dataframe:str):
    train_set = pd.read_parquet(path_train_dataframe)
    train_set =  train_set.drop(['a','redshift', 'mean_FeMassFrac', 'std_FeMassFrac', 'mean_OMassFrac', 'std_OMassFrac'], axis=1)
    
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
    test_set.to_parquet(f'./test_data.parquet')
    
    train_set = train_set[~train_set['Galaxy_name'].isin(test_set['Galaxy_name'])]
    
    #remove the column Galaxy name before passing it to the model
    test_set = test_set[train_set.columns.difference(['Galaxy_name'], sort=False)]
    train_set = train_set[train_set.columns.difference(['Galaxy_name'], sort=False)]
    val_set = val_set[train_set.columns.difference(['Galaxy_name'], sort=False)]
    test_set = torch.from_numpy(np.array(test_set.values, dtype=float))
    val_set = torch.from_numpy(np.array(val_set.values, dtype=float))
    train_set = torch.from_numpy(np.array(train_set.values, dtype=float))
    
    return train_set, val_set

train_set, val_set = load_train_objs('../MW_MH/data/preprocessing_subsample/preprocess_training_set_Galaxy_name_subsample.parquet')


Flow = FreeFormFlow(dim = 2, 
                    cond_dim = 6,
                    hidden_dim = 128,
                    latent_dim = 2,
                    n_SC_layer = 6,
                    beta = 100**2,
                    device = 'cuda:0'
                    )


Flow.train_model(n_epochs = 50,
                 batch_size=1024,
                 optimizer=torch.optim.Adam(Flow.parameters(), lr=2e-4),
                 train_set=train_set,
                 val_set=val_set)

