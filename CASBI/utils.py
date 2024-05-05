import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import scipy.stats as stats
from scipy.stats import gaussian_kde
from scipy.special import kl_div
from scipy.spatial.distance import jensenshannon as js_div
from multiprocessing import Pool
import pynbody as pb

import os
import subprocess
import re
from tqdm.notebook import tqdm
from .generator import *

import ndtest


"""
GENERATION OF THE FILEs OF OBSERVATIONS AND PARAMETERS FOR THE TRAINING SET
In this section we define the functions to extract the parameters and observables from the simulation snapshots and save them in .npz files.
"""
def extract_parameter_array(sim_path='str', file_path='str') -> None:
    """
    Extract the parameters and observables from the path. Checks all the possible errors and if one is found it is saved as an 'error_file'.  
    If no stars were formed in the snapshot, the function dosen't save any file.
    Two .npz files are returned, one with the parameters and another with the observables.
    In order to load the parameters values use the common way of accessing numpy array in .npz file, for example: np.load('file.npz')['star_mass'].
    The parameters that are extracted are: star_mass, gas_mass, dm_mass, infall_time, redshift, a, chemical_mean and chemical_std.
    The observables that are extracted are: [Fe/H], [O/Fe], refered to as 'feh' and 'ofe'.

    Parameters
    ----------
    sim_path : str 
        Path to the simulation snapshot. The path should end with 'simulation_name.snapshot_number' and it is used to create the name of the .npz files.
    file_path : str
        Path to the folder where the file will be saved. The file is a .npz file with parameters and observables stored in it.

    Returns
    -------
    file : .npz array
        The file is save in the folder '/file_path/name_file_parameters.npz'. 
        The parameters are:
        file['star_mass'] : float
            Total mass of the formed stars in the snapshot
        file['gas_mass'] : float
            Total mass of the gas in the snapshot
        file['dm_mass'] : float
            Total mass of the dark matter in the snapshot
        file['infall_time'] : float
            Time at which the snapshot was taken in Gyr
        file['redshift'] : float
            Redshift at which the snapshot was taken
        file['a'] : float
            Scale factor at which the snapshot was taken
        file['chemical_mean'] : np.array
            Array with the mean of metals, FeMassFrac and OxMassFrac in the snapshot
        file['chemical_std'] : np.array
            Array with the standard deviation of metals, FeMassFrac and OxMassFrac in the snapshot

        The observables are:   
        file['feh'] : np.array
            Array with the [Fe/H] of the formed stars in the snapshot
        file['ofe'] : np.array
            Array with the [O/Fe] of the formed stars in the snapshot
    """
    

    #extract the name of the simulation+snapshot_number to create the name of the files to save
    regex = r'[^/]+$'
    name_file = re.search(regex, sim_path).group()
    
    try:
        #check if the file can be loaded
        sim = pb.load(sim_path)
        sim.physical_units()
    except:
        np.savez(file=file_path + name_file + '_load_error.npz', emppty=np.array([0]))
    else:
        try:
            #check if the halos can be loaded
            h = sim.halos()
            h_1 = h[1]
        except:
            print(f'Halo error {name_file}')
            np.savez(file=file_path + name_file + '_halos_error.npz', emppty=np.array([0]))
        else:
            try: 
                mass = h_1.s['mass']
            except:
                print('Dummy halos')
                np.savez(file=file_path + name_file + '_dummy_error.npz', emppty=np.array([0]))           
            else:
                #check if the simualtion has formed stars
                if len(h_1.s['mass']) > 0:
                    
                    file_name = file_path + name_file + '.npz'
                    #PARAMETERS
                    star_mass = np.array(h_1.s['mass'].sum()) #in Msol
                    gas_mass = np.array(h_1.g['mass'].sum())  #in Msol
                    dm_mass = np.array(h_1.dm['mass'].sum())  #in Msol
                    infall_time = np.array(h_1.properties['time'].in_units('Gyr'))
                    redshift = np.array(h_1.properties['z'])
                    a = np.array(h_1.properties['a'])
                    try: 
                        #check if the metals, Iron mass fraction and Oxygen mass fraction mean and std can be extracted
                        mean_metallicity = np.array(h_1.s['metals'].mean())
                        mean_FeMassFrac = np.array(h_1.s['FeMassFrac'].mean())
                        mean_OMassFrac = np.array(h_1.s['OxMassFrac'].mean())
                        std_metallicity = np.array(h_1.s['metals'].std())
                        std_FeMassFrac = np.array(h_1.s['FeMassFrac'].std())
                        std_OMassFrac = np.array(h_1.s['OxMassFrac'].std())
                        
                    except:
                        np.savez(file=file_path + name_file + '_ZMassFracc_error.npz', emppty=np.array([0]))
                    else:
                        #OBSERVABLE
                        try:
                            #check if the [Fe/H] and [O/Fe] can be extracted
                            feh = h_1.s['feh']
                            ofe = h_1.s['ofe']
                        except:
                            np.savez(file=file_path + name_file + '_FeO_error.npz', emppty=np.array([0]))
                        else:
                            np.savez(file=file_name, 
                                     feh=feh, 
                                     ofe=ofe,
                                     star_mass=star_mass, 
                                     gas_mass=gas_mass, 
                                     dm_mass=dm_mass, 
                                     infall_time=infall_time, 
                                     redshift=redshift, 
                                     a=a, 
                                     mean_metallicity=mean_metallicity,
                                     mean_FeMassFrac=mean_FeMassFrac,
                                     mean_OMassFrac=mean_OMassFrac,
                                     std_metallicity=std_metallicity,
                                     std_FeMassFrac=std_FeMassFrac,
                                     std_OMassFrac=std_OMassFrac,
                                     Galaxy_name=name_file,    
                                     )
                else:
                    print('Not formed stars yet')        


def gen_files(sim_path: str, file_path: str) -> None:
    """
    Generate the parameter and observable files for all the given paths, and save them in the 2 separate folders for parameters and observables.
    It is suggested to use the glob library to get all the paths of the snapshots in the simulation like: path = glob.glob('storage/g?.??e??/g?.??e??.0????') 

    Parameters
    ----------
    sim_path : str
        Path to the simulation snapshots. The path should end with 'simulation_name.snapshot_number' and it is used to create the name of the .npz files.
    file_path : str
        Path to the folder where the files will be saved.
    
    Returns
    -------
    None
    
    """                       

    # for p in tqdm(sim_path):
    #     extract_parameter_array(sim_path=p, file_path=file_path)
    pool = Pool(processes=200)
    pool.starmap(extract_parameter_array, zip(sim_path, [file_path]*len(sim_path)))
    

"""        
GENERATION OF THE DATAFRAME 

"""

def rescale(df, mean_and_std_path = str, inverse=False, save = False) -> pd.DataFrame:
    """
    Revert the scaling of the data in the dataframe by adding to each column the mean and multiplying by the standard deviation for the observables.
    The mean and standard deviation are stored in the .parquet file created during the creation of the original dataframe.
    The parameters are not rescaled because the inference would not work correctly.
    
    Parameters:
    df (pandas.DataFrame): 
        The input dataframe to be rescale.
    mean_and_std_path (str): 
        The path to the directory where the .parquet file with the mean and standard deviation of the columns of the dataframe will be saved if save=True.
        Otherwise, if the is already been created, it the path to the file itself.
    inverse (bool):
        If True, the rescaling is reverted.
    save (bool):
        If True, the mean and standard deviation of the columns of the dataframe are saved in a .parquet file in the mean_and_std_path directory.
        Otherwise, the mean and standard deviation are loaded from the mean_and_std_path file.
        
    Returns:
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
        mean_and_std.to_parquet(mean_and_std_path + 'mean_and_std.parquet')
        df.apply(lambda x: (x.to_numpy() - x.to_numpy().mean()) / x.to_numpy().std(), axis=0)
        
    else:
        mean_and_std = pd.read_parquet(mean_and_std_path)    
        if inverse==True:
            for col in df.columns[:2]:   
                mean = mean_and_std.loc[0, f'mean_{col}'] 
                std  = mean_and_std.loc[0, f'std_{col}'] 
                df[col] = df[col]*std + mean
        else:
            for col in df.columns[:2]:   
                mean = mean_and_std.loc[0, f'mean_{col}'] 
                std  = mean_and_std.loc[0, f'std_{col}'] 
                df[col] = (df[col] - mean) / std
    return df


def load_data(file_path, mass_cut=6*1e9, min_n_star=float, min_feh=float, min_ofe=float, n_subsamples = 500):
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
    np.savez(file=f'{preprocess_dir}preprocess_file', Galaxy_Mass=Galaxy_Mass, Number_Star=Number_Star, FeH=FeH, OFe=OFe)
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
    df.to_parquet(dataframe_path + 'dataframe.parquet')
    
    return df 

"""
================================================================================
TRAIN, VALIDATION AND TEST 
================================================================================


"""
def get_even_space_sample(df_mass_masked):
    '''
    Given a dataframe of galaxy in a range of mass, it returns 10 equally infall time spaced samples  
    '''
    len_infall_time = len(df_mass_masked['infall_time'].unique())
    index_val_time = np.linspace(0, len_infall_time-1, 10)
    time = np.sort(df_mass_masked['infall_time'].unique())[index_val_time.astype(int)]
    df_time = pd.DataFrame(columns=df_mass_masked.columns)
    for t in time:
        temp = df_mass_masked[df_mass_masked['infall_time']==t]
        galaxy_temp = temp.sample(1)['Galaxy_name'].values[0]
        df_time = pd.concat((df_time, df_mass_masked[df_mass_masked['Galaxy_name']==galaxy_temp]) )

    return df_time

def load_train_objs(df_path:str, train_path:str, val_path:str, test_path:str) -> None:
    """
    Load and preprocess training data.

    Parameters:
    - df_path (str): Path to the input data file.
    - train_path (str): Path to save the preprocessed training data.
    - val_path (str): Path to save the validation data.
    - test_path (str): Path to save the test data.

    Returns:
    None
    """
    train_set = pd.read_parquet(df_path) 
    low_percentile_mass, high_percentile_mass = np.percentile(train_set['star_log10mass'], 25), np.percentile(train_set['star_log10mass'], 75)
    low_mass = get_even_space_sample(train_set[train_set['star_log10mass']<=low_percentile_mass])
    intermediate_mass = get_even_space_sample(train_set[(train_set['star_log10mass']>low_percentile_mass) & (train_set['star_log10mass']<high_percentile_mass)])
    high_mass = get_even_space_sample(train_set[train_set['star_log10mass']>=high_percentile_mass])
    val_set = pd.concat([low_mass, intermediate_mass, high_mass])
    
    train_set = train_set[~train_set['Galaxy_name'].isin(val_set['Galaxy_name'])]
    
    low_percentile_mass, high_percentile_mass = np.percentile(train_set['star_log10mass'], 25), np.percentile(train_set['star_log10mass'], 75)
    low_mass = get_even_space_sample(train_set[train_set['star_log10mass']<=low_percentile_mass])
    intermediate_mass = get_even_space_sample(train_set[(train_set['star_log10mass']>low_percentile_mass) & (train_set['star_log10mass']<high_percentile_mass)])
    high_mass = get_even_space_sample(train_set[train_set['star_log10mass']>=high_percentile_mass])
    test_set = pd.concat([low_mass, intermediate_mass, high_mass])
    test_set.to_parquet(test_path)
    
    train_set = train_set[~train_set['Galaxy_name'].isin(test_set['Galaxy_name'])]
    print('finish prepare data')

"""
================================================================================
METRIC TEST
================================================================================


"""


def create_sample_df(df_to_sample, Flow, inverse_rescale_file):
    
    other_cols = df_to_sample.columns.difference(['Galaxy_name'])
    df = pd.DataFrame(columns=df_to_sample.columns)
    with torch.no_grad():
        Flow.eval()
        for galaxy in sorted(df_to_sample['Galaxy_name'].unique()):    
            galaxy_data = df_to_sample[df_to_sample['Galaxy_name']==galaxy]
            generated_data = Flow.sample_Flow(len(galaxy_data), galaxy_data[other_cols].values[0,2:], inverse_rescale_file=inverse_rescale_file).cpu().detach()
            generated_data = pd.DataFrame(generated_data.numpy(), columns=['feh', 'ofe'])
            generated_data['Galaxy_name'] = galaxy
            for col in galaxy_data.columns[2:-1]:
                generated_data[col] = galaxy_data[col].values[0]
            df = pd.concat([df, generated_data])
    return df

    
def kl_divergence(p, q):
 return sum(p[i] * np.log2(p[i]/q[i]) for i in range(len(p)))

def get_test_metric(test_df:pd.DataFrame, df_sample, model:torch.nn.Module, inverse=False, inverse_rescale_file=None):
    """
    Return the KL and JS divergence between the KDE of the true data and the generated data for the test set.
    
    Parameters
    ----------
    test_df : pd.DataFrame
        The test set dataframe.
    
    model: torch.nn.Module
        The model used to generate the data.
    
    Returns
    -------
    kl_div_value : float
        The KL divergence between the KDE of the true data and the generated data.
    js_div_value : float
        The JS divergence between the KDE of the true data and the generated data.
    """
    
    bad_column = ['Galaxy_name']
    other_columns = test_df.columns.difference(bad_column, sort=False)
    
    kl_div_all = np.zeros(len(test_df['Galaxy_name'].unique()))
    js_div_all = np.zeros(len(test_df['Galaxy_name'].unique()))
    D = np.zeros(len(test_df['Galaxy_name'].unique()))
    i = 0
    for galaxy in tqdm(sorted(test_df['Galaxy_name'].unique())):
        galaxy_data = test_df[test_df['Galaxy_name']==galaxy]
        x = galaxy_data['feh']
        y = galaxy_data['ofe']
        kde = gaussian_kde(np.vstack([x, y]))
        kde_value = kde.evaluate(np.vstack([x.to_numpy(), y.to_numpy()]))
        flow_pdf  = model.get_pdf(galaxy_data.values[:, :2].astype(float), galaxy_data[other_columns].values[0, 2:].astype(float), inverse=inverse, inverse_rescale_file=inverse_rescale_file)
        kl_div_all[i] = kl_divergence(kde_value, flow_pdf)
        js_div_all[i]  = js_div(kde_value, flow_pdf)
        
        sample_data = df_sample[df_sample['Galaxy_name']==galaxy]
        x_s = sample_data['feh']
        y_s = sample_data['ofe']
        P, D[i] = ndtest.ks2d2s(x_s.to_numpy(), y_s.to_numpy(), x.to_numpy(), y.to_numpy(), extra=True)
        i+=1
    
    kl_div_mean = np.mean(kl_div_all)
    js_div_mean = np.mean(js_div_all)
    D_mean = np.mean(D)
    return kl_div_mean, js_div_mean, D_mean

"""
================================================================================
PLOT FUNCTION
================================================================================


"""
def custom_kde_plot(test_df: pd.DataFrame, df_sample, Flow, inverse_rescale_file, nll: float, kl_mean: float, js_mean:float, D_mean:float, levels=5,):
    """
    
    
    
    """
    test_df['Data'] = 'test'
    df_sample['Data'] = 'sample'
    df_joinplot = pd.concat([test_df, df_sample])
    
    fig, ax = plt.subplots(2, 2, figsize=(6, 6), 
                        gridspec_kw={"height_ratios": (.15, .85), 
                                    "width_ratios": (.85, .15)})
    ax[0, 1].remove()

    colors = [['red'], ['black']]
    patches = []
    
    for i, data_type in enumerate(['test', 'sample']):
        # plot kde for data_type in ['simulation', 'generated']
        x = df_joinplot['feh'][df_joinplot['Data'] == data_type]
        y = df_joinplot['ofe'][df_joinplot['Data'] == data_type]
        
        if data_type == 'test':
            x_t = x
            y_t = y
            kde = gaussian_kde(np.vstack([x_t, y_t]))
            kde_value = kde.evaluate(np.vstack([x_t.to_numpy(), y_t.to_numpy()]))
            other_columns = test_df.columns.difference(['Galaxy_name', 'Data'], sort=False)
            flow_pdf  = Flow.get_pdf(df_sample[other_columns].values[:, :2].astype(float), df_sample[other_columns].values[0, 2:].astype(float), inverse=False, inverse_rescale_file=inverse_rescale_file)
            kl_div_galaxy = kl_divergence(kde_value, flow_pdf)
            js_div_galaxy  = js_div(kde_value, flow_pdf)
            
            
        if data_type == 'sample':
            P, D_galaxy = ndtest.ks2d2s(x_t.to_numpy(), y_t.to_numpy(), x.to_numpy(), y.to_numpy(), extra=True)
    
        # Calculate 2D KDE
        kde = gaussian_kde(np.vstack([x, y]))

        # Create a grid of points for which to evaluate the KDE
        x_d = np.linspace(min(x), max(x), 100)
        y_d = np.linspace(min(y), max(y), 100)
        X, Y = np.meshgrid(x_d, y_d)
        Z = kde.evaluate(np.vstack([X.ravel(), Y.ravel()]))

        # Plot 2D KDE
        contour = ax[1, 0].contour(X, Y, Z.reshape(X.shape), levels=levels,  alpha=0.7, colors=colors[i])

        # Create a patch for the legend
        patches.append(mpatches.Patch(color=contour.collections[0].get_edgecolor(), label=data_type))
        
        # Calculate densities
        kde_x = gaussian_kde(x, )
        kde_y = gaussian_kde(y, )

        # Create an array of values for which to evaluate the KDE
        x_d = np.linspace(min(x), max(x), 1000)
        y_d = np.linspace(min(y), max(y), 1000)

        # Plot KDEs on the marginals
        ax[0, 0].plot(x_d, kde_x(x_d), color=colors[i][0])
        ax[1, 1].plot(kde_y(y_d), y_d, color=colors[i][0])

        # Remove labels from the marginal axes
        ax[0, 0].set_xticks([])
        ax[1, 1].set_yticks([])
        # ax[1, 0].text(ax[1,0].get_xlim()[0], 0.6, galaxy)
        

        # Set labels on the joint plot
        ax[1, 0].set_xlabel('[Fe/H]')
        ax[1, 0].set_ylabel('[O/Fe]')
        
        
        # Add the legend
        ax[1, 0].legend(handles=patches, loc='lower left')
        if data_type == 'sample':
            text_to_show = f'Galaxy: {df_joinplot["Galaxy_name"].unique()[0]} \n nll: {nll[0]:.2f} \n  kl_mean:{kl_mean:.2f}, kl_g:{kl_div_galaxy:.2f} \n js_mean:{js_mean:.2f}, js_g:{js_div_galaxy:.2f} \n D_mean: {D_mean:.2f},  D_g:{D_galaxy:.2f} '
            ax[1, 0].text(0, 1, text_to_show, verticalalignment='top', horizontalalignment='left', transform=ax[1, 0].transAxes)


"""
====================
DISTRIBUTED TRAINING 
====================
"""
def distributed_training(path_train_dataframe: str,
                         test_and_nll_path: str,
                         world_size: int, 
                         number_of_epochs:int, 
                         snapshot_path='snapshot/snapshot.pth',):
    """
    Function to lunch the bash script torchrun to train the model in a distributed way.
    The model is saved in the snapshot.ph file.
    The tensorboard logs are saved in the folder 'runs'.
    
    Parameters
    ----------
    world_size : int
        Number of nodes to use for the training.
    number_of_epochs : int
        Number of epochs to train the model.
    
    Returns
    -------
    None
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    script_path = os.path.join(script_dir, "distributed_training.py")
    command = f"torchrun --standalone --nproc_per_node={world_size} {script_path} {path_train_dataframe} {test_and_nll_path} {number_of_epochs} {snapshot_path}"
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Script execution failed with error: {e.returncode}")
