from ili.dataloaders import StaticNumpyLoader, TorchLoader, NumpyLoader
from ili.dataloaders import TorchLoader
from ili.inference import InferenceRunner

def run_inference(train_yaml_path:str, data_yaml_path:str):
    all_loader = StaticNumpyLoader.from_config(data_yaml_path)
    runner = InferenceRunner.from_config(train_yaml_path)
    runner(loader=all_loader)