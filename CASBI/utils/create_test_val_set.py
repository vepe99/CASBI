import numpy as np
import pandas as pd

"""
==========================
TRAIN, VALIDATION AND TEST 
==========================
Create the training, validation and test set for training the model. The test and validation galaxies are in the following way:
divide the galaxies in 3 groups of mass, low (mass < 25 percentile), intermediate (25 percentile < mass < 75 percentile) and high mass (mass > 75 percentile). 
Then, for each group, select N galaxies equally spaced in infall time.

"""
def get_even_space_sample(df_mass_masked, N=10):
    '''
    Given a dataframe of galaxy in a range of mass, it returns N equally infall time spaced samples  
    
    Parameters
    ----------
    df_mass_masked : pd.DataFrame
        Dataframe with the galaxies in a range of mass
    N : int
        Number of galaxies equally spaced in time to be selected
    
    Returns
    -------
    df_time : pd.DataFrame
        Dataframe with the selected galaxies
    '''
    len_infall_time = len(df_mass_masked['infall_time'].unique())
    index_val_time = np.linspace(0, len_infall_time-1, N)
    time = np.sort(df_mass_masked['infall_time'].unique())[index_val_time.astype(int)]
    for i, t in enumerate(time):
        if i == 0:
            df_time = df_mass_masked[df_mass_masked['infall_time']==t]
        else:
            df_time = pd.concat((df_time, df_mass_masked[df_mass_masked['infall_time']==t]) )
            
    return df_time

def load_train_objs(df_path:str, test_path:str, train_path:None, val_path:None ) -> None:
    """
    Create training, validation and test dataset. The test set is saved in a parquet file in the path test_path.
    The validation set is saved in a parquet file in the path val_path. The training set is saved in a parquet file in the path train_path.
    Parameters:
    - df_path (str): Path to the train dataframe, in a .parquet file.
    - train_path (str): Path to save the preprocessed training data.
    - val_path (str): Path to save the validation data.
    - test_path (str): Path to save the test data.

    Returns:
    train_set, val_set, test_set: pd.DataFrame
    """
    train_set = pd.read_parquet(df_path) 
    
    ### create validation set
    low_percentile_mass, high_percentile_mass = np.percentile(train_set['star_log10mass'], 25), np.percentile(train_set['star_log10mass'], 75)
    low_mass = get_even_space_sample(train_set[train_set['star_log10mass']<=low_percentile_mass])
    intermediate_mass = get_even_space_sample(train_set[(train_set['star_log10mass']>low_percentile_mass) & (train_set['star_log10mass']<high_percentile_mass)])
    high_mass = get_even_space_sample(train_set[train_set['star_log10mass']>=high_percentile_mass])
    val_set = pd.concat([low_mass, intermediate_mass, high_mass])
    if val_path is not None:
        val_set.to_parquet(val_path)
    
    ### remove validation galaxies from train set
    train_set = train_set[~train_set['Galaxy_name'].isin(val_set['Galaxy_name'])]
    
    ### create test set
    low_percentile_mass, high_percentile_mass = np.percentile(train_set['star_log10mass'], 25), np.percentile(train_set['star_log10mass'], 75)
    low_mass = get_even_space_sample(train_set[train_set['star_log10mass']<=low_percentile_mass])
    intermediate_mass = get_even_space_sample(train_set[(train_set['star_log10mass']>low_percentile_mass) & (train_set['star_log10mass']<high_percentile_mass)])
    high_mass = get_even_space_sample(train_set[train_set['star_log10mass']>=high_percentile_mass])
    test_set = pd.concat([low_mass, intermediate_mass, high_mass])
    
    ### save the training set 
    if test_set is not None:
        test_set.to_parquet(test_path)
    
    ### remove the test galaxies from the train set and saves it
    train_set = train_set[~train_set['Galaxy_name'].isin(test_set['Galaxy_name'])]
    if train_path is not None:
        train_set.to_parquet(train_path)
    print('finish prepare data')

    return train_set, val_set, test_set