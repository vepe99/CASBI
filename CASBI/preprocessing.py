import numpy as np
import pandas as pd
import pynbody as pb

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
import logging
import sys
import os
import re 
from tqdm import tqdm

# Configure logging to suppress messages from pynbody
logging.getLogger('pynbody').setLevel(logging.CRITICAL)

"""
===========================================================================
GENERATION OF THE FILEs OF OBSERVATIONS AND PARAMETERS FOR THE TRAINING SET
===========================================================================
Functions to extract the parameters and observables from the simulation snapshots and save them in .npz files.
"""


def extract_parameter_array(sim_path='str', file_path='str') -> None:
    """
    Extract the parameters and observables from the path. Checks all the possible errors and if one is found it is saved as an 'error_file'.  
    If no stars were formed in the snapshot, the function dosen't save any file. Two .npz files are returned, one with the parameters and another with the observables.
    In order to load the parameters values use the common way of accessing numpy array in .npz file, for example: np.load('file.npz')['star_mass'].
    The parameters that are extracted are: star_mass, infall_time.
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
        file['infall_time'] : float
        Time at which the snapshot was taken in Gyr
        
        The observables are:   
        file['feh'] : np.array
        Array with the [Fe/H] of the formed stars in the snapshot
        file['ofe'] : np.array
        Array with the [O/Fe] of the formed stars in the snapshot
    """
    

    #extract the name of the simulation+snapshot_number to create the name of the files to save
    regex = r'[^/]+$'
    name_file = re.search(regex, sim_path).group()
    
    # Redirect stderr to suppress error messages
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    
    error_dataframe = pd.DataFrame(columns=['Galaxy_name', 'error'])
    
    try:
        #check if the file can be loaded
        sim = pb.load(sim_path)
        sim.physical_units()
    except:
        error_dataframe = error_dataframe.append({'Galaxy_name': name_file, 'error': 'load'}, ignore_index=True)
        np.savez(file=os.path.join(file_path, name_file+'_load_error.npz'), emppty=np.array([0]))
    else:
        try:
            #check if the halos can be loaded
            h = sim.halos()
            h_1 = h[1]
        except:
            error_dataframe = error_dataframe.append({'Galaxy_name': name_file, 'error': 'halos'}, ignore_index=True)
            np.savez(file=file_path + name_file + '_halos_error.npz', emppty=np.array([0]))
        else:
            try: 
                mass = h_1.s['mass']
            except:
                error_dataframe = error_dataframe.append({'Galaxy_name': name_file, 'error': 'mass'}, ignore_index=True)
                np.savez(file=os.path.join(file_path, name_file+'_mass_error.npz'), emppty=np.array([0]))           
            else:
                #check if the simualtion has formed stars
                if len(h_1.s['mass']) > 0:
                    
                    file_name = file_path + name_file + '.npz'
                    #PARAMETERS
                    star_mass = np.array(h_1.s['mass'].sum()) #in Msol
                    infall_time = np.array(h_1.properties['time'].in_units('Gyr'))
                    try:
                        #check if the [Fe/H] and [O/Fe] can be extracted
                        feh = h_1.s['feh']
                        ofe = h_1.s['ofe']
                    except:
                        error_dataframe = error_dataframe.append({'Galaxy_name': name_file, 'error': 'chemical'}, ignore_index=True)
                        np.savez(file=os.path.join(file_path, name_file+'_FeO_error.npz'), emppty=np.array([0]))
                    else:
                        np.savez(file=file_name, 
                                    feh=feh, 
                                    ofe=ofe,
                                    star_mass=star_mass, 
                                    infall_time=infall_time, 
                                    Galaxy_name=name_file,    
                                    )
                else:
                    error_dataframe = error_dataframe.append({'Galaxy_name': name_file, 'error': 'no_stars'}, ignore_index=True)
                    print('Not formed stars yet')      
    finally:
        # Restore stderr
        sys.stderr.close()
        sys.stderr = original_stderr  
    
    return error_dataframe


def gen_files(sim_path: str, file_path: str) -> None:
    """
    Generate the parameter and observable files for all the given paths, and save them in the 2 separate folders for parameters and observables.
    It is suggested to use the glob library to get all the paths of the snapshots in the simulation like: path = glob.glob('storage/g?.??e??/g?.??e??.0????') 
    Saves also a dataframe with the errors that occurred during the extraction of the parameters and observables, in the same directory as the files.
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
    with Pool(processes=cpu_count()) as pool:                       
        df_list = pool.starmap(extract_parameter_array, zip(sim_path, [file_path]*len(sim_path)))
    
    error_dataframe = pd.concat(df_list, ignore_index=True)
    error_dataframe.to_parquet(os.path.join(file_path, 'error_dataframe.parquet'))
    
"""
===========================================================================
GENERATION DATAFRAME
===========================================================================
The dataframe with information on parameters and observables for all the galaxy available in the simulation.
The preprocess is used to cut numerical errors and outliers (especially in the chemical plane)
"""


def preprocess(file_dir:str,  preprocess_dir:str) -> None:
    """
    Save the necessary files to preprocess the data for the training set. It saves aggregated information of Galaxy Mass, Number of stars, [Fe/H] and [O/Fe] in the preprocess_dir.
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
    Galaxy_infall_time = []
    FeH = []
    OFe = []
    
    for galaxy in tqdm(os.listdir(file_dir)):
        if not("error" in galaxy): 
            path = os.path.join(file_dir,galaxy) 
            mass = np.load(path)['star_mass']
            time = np.load(path)['infall_time']
            Galaxy_Mass.append(float(mass))    
            Galaxy_infall_time.append(float(time))
            
            #we need a general pictures of the entire distribution of [Fe/H] and [O/Fe] to cut the outliers
            feh = np.load(path)['feh']
            ofe = np.load(path)['ofe']
            for f, o in zip(feh, ofe):
                FeH.append(f)    
                OFe.append(o)
            
    Galaxy_Mass = np.array(Galaxy_Mass)
    Galaxy_infall_time = np.array(Galaxy_infall_time)
    np.savez(file=os.path.join(preprocess_dir, 'preprocess_file'), Galaxy_Mass=Galaxy_Mass, Galaxy_infall_time=Galaxy_infall_time, FeH=FeH, OFe=OFe)
    return f'{preprocess_dir}preprocess_file.npz'


def load_data(file_path):
    """
    Load the data from the file_path and return a pandas dataframe with the data. This function is then distributed in CASBI.preprocessing.gen_dataframe function
    
    Parameters
    ----------
    file_path : str
        Path to the file with the parameters and observables.

    Returns
    -------
    df_temp : pandas.DataFrame
        The dataframe with the data from the file_path.
    """
    properties = ['star_mass', 'infall_time', 'Galaxy_name', 'max_feh', 'max_ofe']
    data = [np.load(file_path)[prop].item() for prop in properties[:3]]
    #get the maximum of feh and ofe
    data.append(np.load(file_path)['feh'].max())
    data.append(np.load(file_path)['ofe'].max())

    df_temp = pd.DataFrame(columns = properties)
    df_temp.loc[0] = data

    return df_temp

def gen_dataframe(file_dir: str, dataframe_path: str) -> None:
    """
    Genereate the dataframe used for the sampling process in the CASBI.template_library class
    
    Parameters
    ----------
    file_dir : str
        Path to the folder where the files with the parameters and observables are saved.
    
    dataframe_path : str
        Path to the folder where the dataframe will be saved
    
    Returns
    -------
    df : pandas.DataFrame
        The dataframe with the data from the file_dir.
    """
    
    #access all the file created by preprocessing.gen_files
    all_files = sorted(os.listdir(file_dir))
    regex = r'^(?!.*error)'
    file_path = [os.path.join(file_dir,path) for path in all_files if re.search(regex, path)]
    
    #distributed the data access
    with Pool(processes=100) as pool:
        df_list = pool.map(load_data, file_path)
    
    #concatenate the dataframes
    df = pd.concat(df_list, ignore_index=True)
    
    #save the dataframe
    df.to_parquet(os.path.join(dataframe_path, 'dataframe.parquet'))

    return df 

