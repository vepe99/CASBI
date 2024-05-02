import numpy as np
import pandas as pd
import scipy.stats as stats
from multiprocessing import Pool

import pynbody as pb

import re
from tqdm.notebook import tqdm


########GENERATION OF THE FILE OBSERVATION AND PARAMETERS FOR THE TRAINING SET########
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

    for p in tqdm(sim_path):
        extract_parameter_array(sim_path=p, file_path=file_path)
        
        
#######GENERATION OF THE DATAFRAME FOR THE TRAINING SET########
def rescale(df, mean_and_std_path = str) -> pd.DataFrame:
    '''
    Rescale the data in the dataframe by removing to each column the mean and dividing by the standard deviation.
    Mean and standard deviation are stored in a .parqet dataframe to revert the normalization during inference

    Parameters:
    df (pandas.DataFrame): The input dataframe to be rescale.
    mean_and_std_path (str): The path to the .parquet file with the mean and standard deviation of the columns of the dataframe.
    
    Returns:
    None
    '''
    columns = []
    for col in df.columns[:-1]:
        columns.append(f'mean_{col}')
        columns.append(f'std_{col}')
    mean_and_std = pd.DataFrame(columns=columns)
    for col in df.columns:
        mean_and_std.loc[0, f'mean_{col}'] = df[col].mean()
        mean_and_std.loc[0, f'std_{col}'] = df[col].std()   
    mean_and_std.to_parquet(mean_and_std_path + '.parquet')    
    return df.apply(lambda x: (x.to_numpy() - x.to_numpy().mean()) / x.to_numpy().std(), axis=0)

def inverse_rescale(df, mean_and_std_file = str) -> pd.DataFrame:
    """
    Revert the scaling of the data in the dataframe by adding to each column the mean and multiplying by the standard deviation for the observables.
    The mean and standard deviation are stored in the .parquet file created during the creation of the original dataframe.
    The parameters are not rescaled because the inference would not work correctly.
    
    Parameters:
    df (pandas.DataFrame): 
        The input dataframe to be rescale.
    mean_and_std_file (str): 
        The path to the .parquet file with the mean and standard deviation of the columns of the dataframe.
        
    Returns:
    df (pandas.DataFrame): 
        The dataframe with the observables data rescaled.

    """
    mean_and_std = pd.read_parquet(mean_and_std_file)
    for col in df.columns[:2]:
        mean = mean_and_std.loc[0, f'mean_{col}'] 
        std  = mean_and_std.loc[0, f'std_{col}']    
        df[col] = df[col]*std + mean
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
            data = np.zeros((n_subsamples, len(components)))
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
            data[:, 14] = [file_array['Galaxy_name'] for i in range(len(ones))]
            
            df_temp = pd.DataFrame(data, columns=components)
            return df_temp
        
def preprocess_setup(file_dir:str,  preprocess_file:str) -> None:
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
    None
    
    """
    Galaxy_Mass = []
    Number_Star = []
    FeH = []
    OFe = []
    
    for galaxy in tqdm(os.listdir(file_dir)):
        if not("error" in galaxy): 
            path = directory + galaxy 
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
    np.savez(file='preprocess_file', Galaxy_Mass=Galaxy_Mass, Number_Star=Number_Star, FeH=FeH, OFe=OFe)

def gen_dataframe(file_dir: str, dataframe_path: str, preprocess_file:str, perc_star=10, perc_feh=0.1, perc_ofe=0.1) -> None:
    min_n_star = np.percentile(np.load(preprocess_file)['Number_Star'], perc_star)
    min_feh    = np.percentile(np.load(preprocess_file)['FeH'], perc_feh)
    min_ofe    = np.percentile(np.load(preprocess_file)['OFe'], perc_ofe) 
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
    df[other_cols] = rescale(df[other_cols]) #nomalization must be then reverted during inference to get the correct results
    df.to_parquet(dataframe_path + '.parquet')


