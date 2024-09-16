import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dask.dataframe as dd
from dask import delayed
import multiprocessing as mp



import torch
import os


if __name__ == "__main__":
    def save_galaxy_data(galaxy_name, df, save_path: str = '/export/data/vgiusepp/data/full_dataframe/galaxy_array/'):
        galaxy_df = df[df['Galaxy_name'] == galaxy_name]
        
        feh = galaxy_df['feh'].values
        ofe = galaxy_df['ofe'].values
        
        np.savez(os.path.join(save_path, galaxy_name), feh=feh, ofe=ofe)
        print(f'Saved {galaxy_name}.npz')

    def process_and_save(data_path):
        # Read the DataFrame using Dask
        ddf = dd.read_parquet(data_path)
        
        # Get unique galaxy names
        unique_galaxy_names = ddf['Galaxy_name'].unique().compute()
        
        # Process each galaxy name in parallel
        dask_tasks = [delayed(save_galaxy_data)(galaxy_name, ddf) for galaxy_name in unique_galaxy_names]
        
        # Compute the tasks
        delayed_tasks = delayed(lambda: [task.compute() for task in dask_tasks])()
        delayed_tasks.compute()


    # Example usage
    data_path = '/export/data/vgiusepp/data/full_dataframe/dataframe/dataframe.parquet'
    process_and_save(data_path)