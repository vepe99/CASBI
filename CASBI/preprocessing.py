import numpy as np
import re 
import pynbody as pb
from multiprocessing import Pool
import logging
import sys
import os

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
            h_0 = h[0]
        except:
            print(f'Halo error {name_file}')
            np.savez(file=file_path + name_file + '_halos_error.npz', emppty=np.array([0]))
        else:
            try: 
                mass = h_0.s['mass']
            except:
                print('Dummy halos')
                np.savez(file=file_path + name_file + '_dummy_error.npz', emppty=np.array([0]))           
            else:
                #check if the simualtion has formed stars
                if len(h_0.s['mass']) > 0:
                    
                    file_name = file_path + name_file + '.npz'
                    #PARAMETERS
                    star_mass = np.array(h_0.s['mass'].sum()) #in Msol
                    infall_time = np.array(h_0.properties['time'].in_units('Gyr'))
                    try:
                        #check if the [Fe/H] and [O/Fe] can be extracted
                        feh = h_0.s['feh']
                        ofe = h_0.s['ofe']
                    except:
                        np.savez(file=file_path + name_file + '_FeO_error.npz', emppty=np.array([0]))
                    else:
                        np.savez(file=file_name, 
                                    feh=feh, 
                                    ofe=ofe,
                                    star_mass=star_mass, 
                                    infall_time=infall_time, 
                                    Galaxy_name=name_file,    
                                    )
                else:
                    print('Not formed stars yet')      
    finally:
        # Restore stderr
        sys.stderr.close()
        sys.stderr = original_stderr  


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
    pool = Pool(processes=200)
    pool.starmap(extract_parameter_array, zip(sim_path, [file_path]*len(sim_path)))
    