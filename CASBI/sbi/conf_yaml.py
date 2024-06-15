#generate trainig and data yaml file
import yaml
import CASBI

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
        "thetafid_file": thetafid_file
    }
    with open(filepath, 'w') as file:
        yaml.dump(data_yaml, file)
        
def generate_training_yaml(filepath:str, output_file:str , hidden_feature:int=100, num_transforms:int=10, model:str='nsf', embedding_net:str='CNN', output_dim:int=128):
    """
    Create the training.yaml file to use as input for the training process. This function create a yaml file that should be customize by the user.

    Args:
        filepath (str): The path where yaml file will be stored.
    """
    training_yaml = {
        "device": "cuda",
        "embedding_net": {
            "args": {
                "input_channel": 1,
                "output_dim": output_dim,
            },
            "class": "ConvNet",
            "module": f"CASBI.sbi.{embedding_net}"
        },
        "model": {
            "backend": "sbi",
            "engine": "NPE",
            "name": f"{output_file}",
            "nets": [
                {
                    "hidden_features": hidden_feature,
                    "model": f"{model}",
                    "num_transforms": num_transforms,
                    "signature": "m1"
                },
                {
                    "hidden_features": hidden_feature,
                    "model": f"{model}",
                    "num_transforms": num_transforms,
                    "signature": "m2"
                },
                {
                    "hidden_features": hidden_feature,
                    "model": f"{model}",
                    "num_transforms": num_transforms,
                    "signature": "m3"
                },
                {
                    "hidden_features": hidden_feature,
                    "model": f"{model}",
                    "num_transforms": num_transforms,
                    "signature": "m4"
                },
            ]
        },
        "out_dir": "./",
        "prior": {
            "args": {
                "high": [30.0],
                "low": [2.0]
            },
            "class": "Uniform",
            "module": "ili.utils"
        },
        "train_args": {
            "learning_rate":  0.00001,
            "stop_after_epochs": 20,
            "training_batch_size": 1024,
            "validation_fraction": 0.2
        }
    }

    with open(filepath, 'w') as file:
        yaml.dump(training_yaml, file)