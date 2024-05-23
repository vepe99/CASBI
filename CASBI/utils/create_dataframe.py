import numpy as np
import pandas as pd
import os
from tqdm.notebook import tqdm 
import re 
from multiprocessing import Pool

"""        
===========================
GENERATION OF THE DATAFRAME 
===========================
Function to preprocess the file obtain from `CASBI.utils.prepare_file` and save the data in a dataframe.
"""

def rescale(df, mean_and_std_path = str, inverse=False, save = False, scale_observations=True, scale_parameters=False) -> pd.DataFrame:
    """
    If save=True the columns of the dataframe are rescaled and the mean and standard deviation of the columns are saved in a .parquet file in the mean_and_std_path directory.
    If save=False, the function applies the inverse rescaling when inverse=True, or apply the same rescaling as when save=True it when inverse=False.
    If scale_observarions=True and scale_parameter=False, only the observables columns are rescaled accordingly to the `inverse` parameter. 
    If scale_observations=True and scale_parameter=True , only the parameters columns are rescaled accordingly to the `inverse` parameter .
    Otherwise all columns will be rescaled accordingly to the `inverse` parameter.
    The parameters are not rescaled because the inference would not work correctly.
    
    Parameters
    -----------
    df (pandas.DataFrame): 
        The input dataframe to be rescale.
    mean_and_std_path (str): 
        The path to the directory where the .parquet file with the mean and standard deviation of the columns of the dataframe will be saved if save=True.
        If save=False, the file is assumed to be already created, and `mean_and_std_path` is the path to the file itself.
    inverse (bool):
        If True, the rescaling is reverted.
    save (bool):
        If True, the mean and standard deviation of the columns of the dataframe are saved in a .parquet file in the mean_and_std_path directory.
        Otherwise, the mean and standard deviation are loaded from the mean_and_std_path file.
        
    Returns
    --------
    df (pandas.DataFrame): 
        The dataframe with the observables data rescaled. If save=True all the columns are rescaled and the mean and standard deviation are saved in the mean_and_std_path directory.
        Otherwise only the observables columns are rescaled.

    """
    if save == True:
        columns = []
        for col in df.columns:
            columns.append(f'mean_{col}')
            columns.append(f'std_{col}')
        mean_and_std = pd.DataFrame(columns=columns)
        mean_and_std.to_parquet(os.path.join(mean_and_std_path, 'mean_and_std.parquet'))
        df.apply(lambda x: (x.to_numpy() - x.to_numpy().mean()) / x.to_numpy().std(), axis=0)
    
    else:
        mean_and_std = pd.read_parquet(mean_and_std_path)    
        if (scale_observations==True) & (scale_parameters==False):    
            if inverse==True:
                for col in df.columns[:2]:  
                    if col != 'Galaxy_name': 
                        mean = mean_and_std.loc[0, f'mean_{col}'] 
                        std  = mean_and_std.loc[0, f'std_{col}'] 
                        df[col] = df[col]*std + mean
            else:
                for col in df.columns[:2]:   
                    if col != 'Galaxy_name':
                        mean = mean_and_std.loc[0, f'mean_{col}'] 
                        std  = mean_and_std.loc[0, f'std_{col}'] 
                        df[col] = (df[col] - mean) / std
                    
        elif (scale_parameters==False) & (scale_observations==True):
            if inverse==True:
                for col in df.columns[2:]:   
                    if col != 'Galaxy_name':
                        mean = mean_and_std.loc[0, f'mean_{col}'] 
                        std  = mean_and_std.loc[0, f'std_{col}'] 
                        df[col] = df[col]*std + mean
            else:
                for col in df.columns[2:]:   
                    if col != 'Galaxy_name':
                        mean = mean_and_std.loc[0, f'mean_{col}'] 
                        std  = mean_and_std.loc[0, f'std_{col}'] 
                        df[col] = (df[col] - mean) / std
        else: 
            if inverse==True:
                for col in df.columns:   
                    if col != 'Galaxy_name':
                        mean = mean_and_std.loc[0, f'mean_{col}'] 
                        std  = mean_and_std.loc[0, f'std_{col}'] 
                        df[col] = df[col]*std + mean
            else:
                for col in df.columns:   
                    if col != 'Galaxy_name':
                        mean = mean_and_std.loc[0, f'mean_{col}'] 
                        std  = mean_and_std.loc[0, f'std_{col}'] 
                        df[col] = (df[col] - mean) / std
    
    return df


def load_data(file_path, mass_cut=6*1e9, min_n_star=float, min_feh=float, min_ofe=float, n_subsamples = 5000):
    """
    Load the data from the file_path and return a pandas dataframe with the data.
    
    Parameters
    ----------
    file_path : str
        Path to the file with the parameters and observables.
    mass_cut : float
        Maximum mass of the stars in the snapshot. If the mass of the stars is greater than this value, the snapshot is not considered.
    min_n_star : float
        Minimum number of stars in the snapshot. If the number of stars is smaller than this value, the snapshot is not considered.
    min_feh : float
        Minimum value of [Fe/H] in the snapshot. If the one star has a value is smaller than min_feh, the star is not considered in order to remove outlier.
    min_ofe : float
        Minimum value of [O/Fe] in the snapshot. If the one star has a value is smaller than min_ofe, the star is not considered in order to remove outlier.
    n_subsamples : int
        Number of subsamples to be taken from the snapshot that are not outliers. 

    Returns
    -------
    df_temp : pandas.DataFrame
        The dataframe with the data from the file_path.
    """
    components = [i.replace('mass', 'log10mass') for i in np.load(file_path).keys()]
    mass = np.load(file_path)['star_mass']
    if mass < mass_cut:
        file_array = np.load(file_path)
        if len(file_array['feh']) > min_n_star:
            l = len([a for a in file_array['feh'][(file_array['feh']>min_feh) & (file_array['ofe']>min_ofe)] ])
            if l < n_subsamples:
                n_subsamples = l
            subsample = np.random.choice(a=range(l), size=n_subsamples, replace=False)
            data = np.zeros((n_subsamples, len(components)-1))
            data[:, 0] = file_array['feh'][(file_array['feh']>min_feh) & (file_array['ofe']>min_ofe)][subsample]
            data[:, 1] = file_array['ofe'][(file_array['feh']>min_feh) & (file_array['ofe']>min_ofe)][subsample]
            ones = np.ones(n_subsamples)
            data[:, 2] = np.log10(file_array['star_mass'])*ones
            data[:, 3] = np.log10(file_array['gas_mass'])*ones
            data[:, 4] = np.log10(file_array['dm_mass'])*ones
            data[:, 5] = file_array['infall_time']*ones
            data[:, 6] = file_array['redshift']*ones
            data[:, 7] = file_array['a']*ones
            data[:, 8] = file_array['mean_metallicity']*ones
            data[:, 9] = file_array['mean_FeMassFrac']*ones
            data[:, 10] = file_array['mean_OMassFrac']*ones
            data[:, 11] = file_array['std_metallicity']*ones
            data[:, 12] = file_array['std_FeMassFrac']*ones
            data[:, 13] = file_array['std_OMassFrac']*ones
             
            df_temp = pd.DataFrame(data, columns=components[:-1])
            df_temp['Galaxy_name'] = file_array['Galaxy_name']
            return df_temp
        
def preprocess_setup(file_dir:str,  preprocess_dir:str) -> None:
    """
    Save the necessary files to preprocess the data for the training set. It savez aggregated information of Galaxy Mass, Number of stars, [Fe/H] and [O/Fe] in the preprocess_dir.
    so that percentile cut can be computed in gen_dataframe funciton
    
    Parameters
    ----------
    file_dir : str
        Path to the folder where the files with the parameters and observables are saved.
    preprocess_dir : str
        Path to the folder where the preprocess information will be saved.
        
    Returns
    -------
    preprocess_file_path: str
        Path to the file with the preprocess information.
    
    """
    Galaxy_Mass = []
    Number_Star = []
    FeH = []
    OFe = []
    
    for galaxy in tqdm(os.listdir(file_dir)):
        if not("error" in galaxy): 
            path = file_dir + galaxy 
            mass = np.load(path)['star_mass']
            number_star = len(np.load(path.replace('parameters', 'observables'))['feh'])
            Galaxy_Mass.append(float(mass))    
            Number_Star.append(number_star)
            
            feh = np.load(path)['feh']
            ofe = np.load(path)['ofe']
            for f, o in zip(feh, ofe):
                FeH.append(f)    
                OFe.append(o)
            
    Galaxy_Mass = np.array(Galaxy_Mass)
    Number_Star = np.array(Number_Star) 
    np.savez(file=os.path.join(preprocess_dir, 'preprocess_file'), Galaxy_Mass=Galaxy_Mass, Number_Star=Number_Star, FeH=FeH, OFe=OFe)
    return f'{preprocess_dir}preprocess_file.npz'


def gen_dataframe(file_dir: str, dataframe_path: str, preprocess_file_path:str, perc_star=10, perc_feh=0.1, perc_ofe=0.1) -> None:
    min_n_star = np.percentile(np.load(preprocess_file_path)['Number_Star'], perc_star)
    min_feh    = np.percentile(np.load(preprocess_file_path)['FeH'], perc_feh)
    min_ofe    = np.percentile(np.load(preprocess_file_path)['OFe'], perc_ofe) 
    mass_cat = 6*1e9
     
    all_files = sorted(os.listdir(file_dir))
    regex = r'^(?!.*error)'
    file_path = [file_dir+path for path in all_files if re.search(regex, path)]
    
    pool = Pool(processes=100)
    items = zip(file_path, [mass_cat]*len(file_path), [min_n_star]*len(file_path), [min_feh]*len(file_path), [min_ofe]*len(file_path))
    df_list = pool.starmap(load_data, items)
    df = pd.concat(df_list, ignore_index=True)
    
    bad_column = 'Galaxy_name'
    other_cols = df.columns.difference([bad_column])    
    df[other_cols] = rescale(df[other_cols], mean_and_std_path=preprocess_file_path.replace('preprocess_file.npz', ''), save=True) #nomalization must be then reverted during inference to get the correct results
    df.to_parquet(os.path.join(dataframe_path, 'dataframe.parquet'))
    
    return df 