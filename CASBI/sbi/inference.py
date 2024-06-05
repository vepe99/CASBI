"""
Wrapper around the ltu-ili packages for the inference task. For now only the StaticNumpyLoader is supported.
"""

from ili.dataloaders import StaticNumpyLoader, TorchLoader, NumpyLoader
from ili.dataloaders import TorchLoader
from ili.inference import InferenceRunner

def run_inference(train_yaml_path:str, data_yaml_path:str):
    """
    Run the inference task using the given yaml files.
    
    Args:
    train_yaml_path (str): Path to the yaml file containing the training configuration.
    data_yaml_path (str): Path to the yaml file containing the data configuration.
    
    """
    all_loader = StaticNumpyLoader.from_config(data_yaml_path)
    runner = InferenceRunner.from_config(train_yaml_path)
    runner(loader=all_loader)