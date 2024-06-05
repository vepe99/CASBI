#generate trainig and data yaml file
import yaml 

import yaml

def generate_data_yaml(filepath:str, in_dir:str, x_file:str='x.npy', theta_file:str='theta.npy', xobs_file:str='x_0.npy', thetafid_file:str='theta_0.npy'):
    """
    Create the data.yaml file to use as input for the StaticNumpyLoader of the ltu-ili package. The data are supposed to be stored in the in_dir directory as .npy files.
    
    Args:
        filepath (str): The path where yaml file will be stored.
        in_dir (str): The directory where the data files are stored.
        x_file (str, optional): The filename of the x data file. Defaults to 'x.npy'.
        theta_file (str, optional): The filename of the theta data file. Defaults to 'theta.npy'.
        xobs_file (str, optional): The filename of the x_0 data file. Defaults to 'x_0.npy'.
        thetafid_file (str, optional): The filename of the theta_0 data file. Defaults to 'theta_0.npy'.
    """
    data_yaml = {
        "in_dir": in_dir,
        "x_file": x_file,
        "theta_file": theta_file,
        "xobs_file": xobs_file,
        "theta_fid": thetafid_file
    }
    with open(filepath, 'w') as file:
        yaml.dump(data_yaml, file)