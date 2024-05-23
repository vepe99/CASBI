import os
import argparse
import numpy as np
from ili.dataloaders import StaticNumpyLoader
from ili.inference import InferenceRunner
from ili.validation import ValidationRunner

if __name__ == '__main__':
    # reload all simulator examples as a dataloader
    all_loader = StaticNumpyLoader.from_config("./data.yaml")

    # train a model to infer x -> theta. save it as toy/posterior.pkl
    runner = InferenceRunner.from_config(
        f"./training.yaml")
    runner(loader=all_loader)