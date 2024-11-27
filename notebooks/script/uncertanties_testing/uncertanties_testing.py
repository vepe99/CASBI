import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
mpl.style.use('../../paper.mcstyle')

import matplotlib.cm as cm
import pandas as pd
import dask.dataframe as dd
from dask import delayed
from multiprocessing import cpu_count
from multiprocessing import Pool
from sklearn.neighbors import NearestNeighbors
from ili.validation.metrics import PosteriorCoverage

from tqdm import tqdm
import torch
import pickle
import CASBI.inference as inference
from CASBI.utils.create_template_library import TemplateLibrary

gpu_index = 6  # replace with your desired GPU index
torch.cuda.set_device(gpu_index)
device = f"cuda:{gpu_index}"

#paper column width is 397.48499pt
pt = 1/72.27
column_width = 397.48499*pt
aspect_ratio = 1.61803398875
my_width = column_width
my_height = my_width/aspect_ratio

def run_with_retry(num_samples):
    while True:
        try:
            metric = PosteriorCoverage(
                num_samples=num_samples, sample_method='direct',
                labels=[rf'$\log_{{10}}(M_{{s}}) [M_{{\odot}}]$', rf'$\log_{{10}}(\tau) [Gyr]$'], plot_list=plot_hist
            )

            fig = metric(
                posterior=posterior_ensamble,
                x=x_test, theta=params_test[:, :2]
            )

            fig[0].savefig(f'./coverage_{sigma}.pdf')
            fig[1].savefig(f'./predictions_{sigma}.pdf')
            fig[2].savefig(f'./accuracy_{sigma}.pdf')
            fig[3].savefig(f'./tarp_{sigma}.pdf')
            break  # Exit the loop if no exception is raised
        except Exception as e:
            print(f"Error encountered: {e}. Retrying with num_samples={num_samples + 1}")
            num_samples += 1

if __name__ == "__main__":
    galaxy_array_path = '/export/data/vgiusepp/data/full_dataframe/galaxy_array/'
    dataframe_path = '/export/data/vgiusepp/data/full_dataframe/dataframe/dataframe.parquet'
    for sigma in tqdm([0.0, 0.01, 0.02, 0.04, 0.06, 0.13]):
        template_library = TemplateLibrary(galaxy_array_path=galaxy_array_path, dataframe_path=dataframe_path, sigma=sigma)    
        template_library.gen_libary(N_test=100, N_train=1000)
        
        # Save dictionary to a pickle file
        with open(f'./test_set/test_set_{sigma}.pkl', 'wb') as json_file:
            pickle.dump(template_library.test_galaxies, json_file)
        
        x_train, params_train, x_test, params_test = template_library.get_inference_input()
        posterior_ensamble, summaries = inference.train_inference(x=x_train, theta=params_train, learning_rate=1e-4, output_dir=f'./posterior/posterior_{sigma}')
        
        # plot train/validation loss
        fig, ax = plt.subplots(1, 1, figsize=(my_width,my_height))
        c = list(mcolors.TABLEAU_COLORS)
        for i, m in enumerate(summaries):
            ax.plot(m['training_log_probs'], ls='-', label=f"{i}_train", c=c[i])
            ax.plot(m['validation_log_probs'], ls='--', label=f"{i}_val", c=c[i])
        ax.set_xlim(0)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Log probability')
        ax.legend()
        fig.savefig(f'./train_val_loss_{sigma}.png')
        
        plot_hist = ["coverage", "histogram", "predictions", "tarp"]
        initial_num_samples = 2001
        # Run the function with retry logic
        run_with_retry(initial_num_samples)

        