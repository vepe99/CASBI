import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import  gaussian_kde
from scipy.spatial.distance import jensenshannon as js_div

from CASBI.generator.fff.fff_model import FreeFormFlow
from CASBI.generator.nf.nf_model import NF_condGLOW
from CASBI.utils.metrics import ks2d2s, kl_divergence

import torch

"""
================================================================================
PLOTS FUNCTION
================================================================================


"""
def custom_kde_plot(test_df: pd.DataFrame, df_sample:pd.DataFrame, model:torch.nn.Module, kl_mean: float, js_mean:float, D_mean:float, levels=5,):
    """
    Visualize the 2D KDE of the test data and the sample data, report also the KL, JS and D statics of the single data generation and the mean over the sets.
    Both marginals and the joint plot are shown.
    
    Parameters
    ----------
    
    
    Returns
    -------
    
    
    """
    if 'Data' not in test_df.columns:
        test_df.insert(len(test_df.columns), 'Data', 'NIHAO')
    if 'Data' not in df_sample.columns:
        df_sample.insert(len(df_sample.columns), 'Data', 'Generated')
    df_joinplot = pd.concat([test_df, df_sample])
    
    fig, ax = plt.subplots(2, 2, figsize=(6, 6), 
                        gridspec_kw={"height_ratios": (.15, .85), 
                                    "width_ratios": (.85, .15)})
    ax[0, 1].remove()

    colors = [['red'], ['black']]
    patches = []
    with torch.no_grad():
        model.eval()
        for i, data_type in enumerate(['NIHAO', 'Generated']):
            # plot kde for data_type in ['simulation', 'generated']
            x = df_joinplot['feh'][df_joinplot['Data'] == data_type]
            y = df_joinplot['ofe'][df_joinplot['Data'] == data_type]
            
            if data_type == 'NIHAO':
                x_t = x
                y_t = y
                kde = gaussian_kde(np.vstack([x_t, y_t]))
                kde_value = kde.evaluate(np.vstack([x_t.to_numpy(), y_t.to_numpy()]))
                other_columns = test_df.columns.difference(['Galaxy_name', 'Data'], sort=False)
                if isinstance(model, NF_condGLOW): 
                    flow_pdf  = model.get_pdf(torch.from_numpy(test_df.values[:, :2].astype(float)), torch.from_numpy(test_df[other_columns].values[0, 2:].astype(float)))
                elif isinstance(model, FreeFormFlow):
                    galaxy_observable = torch.from_numpy(test_df.values[:, :2].astype(float)).to(torch.float32).to(model.device)
                    galaxy_condition = torch.from_numpy(test_df[other_columns].values[:, 2:].astype(float)).to(torch.float32).to(model.device)
                    _, flow_logprob = model.log_prob(galaxy_observable, galaxy_condition)
                    flow_pdf = np.exp(flow_logprob.cpu())
                kl_div_galaxy = kl_divergence(kde_value, flow_pdf)
                js_div_galaxy  = js_div(kde_value, flow_pdf)
                
                
            if data_type == 'Generated':
                P, D_galaxy = ks2d2s(x_t.to_numpy(), y_t.to_numpy(), x.to_numpy(), y.to_numpy(), extra=True)
        
            # Calculate 2D KDE
            kde = gaussian_kde(np.vstack([x, y]))

            # Create a grid of points for which to evaluate the KDE
            x_d = np.linspace(min(x), max(x), 100)
            y_d = np.linspace(min(y), max(y), 100)
            X, Y = np.meshgrid(x_d, y_d)
            Z = kde.evaluate(np.vstack([X.ravel(), Y.ravel()]))

            # Plot 2D KDE
            contour = ax[1, 0].contour(X, Y, Z.reshape(X.shape), levels=levels,  alpha=0.7, colors=colors[i])

            # Create a patch for the legend
            patches.append(mpatches.Patch(color=contour.collections[0].get_edgecolor(), label=data_type))
            
            # Calculate densities
            kde_x = gaussian_kde(x, )
            kde_y = gaussian_kde(y, )

            # Create an array of values for which to evaluate the KDE
            x_d = np.linspace(min(x), max(x), 1000)
            y_d = np.linspace(min(y), max(y), 1000)

            # Plot KDEs on the marginals
            ax[0, 0].plot(x_d, kde_x(x_d), color=colors[i][0])
            ax[1, 1].plot(kde_y(y_d), y_d, color=colors[i][0])

            # Remove labels from the marginal axes
            ax[0, 0].set_xticks([])
            ax[1, 1].set_yticks([])
            # ax[1, 0].text(ax[1,0].get_xlim()[0], 0.6, galaxy)
            

            # Set labels on the joint plot
            ax[1, 0].set_xlabel('[Fe/H]')
            ax[1, 0].set_ylabel('[O/Fe]')
            
            
            # Add the legend
            ax[1, 0].legend(title= f'{df_joinplot["Galaxy_name"].unique()[0]}', handles=patches, loc='lower left')
            if data_type == 'sample':
                text_to_show = f'$\overline{{KL}}$:{kl_mean:.1e}, $KL_g$:{kl_div_galaxy:.1e}' + '\n' + f'$\overline{{JS}}$:{js_mean:.2f}, $JS_g$:{js_div_galaxy:.2f}'+ '\n' + f' $\overline{{D}}$: {D_mean:.2f},  $D_g$:{D_galaxy:.2f} '
                ax[1, 0].text(0.05, 0.95, text_to_show, verticalalignment='top', horizontalalignment='left', transform=ax[1, 0].transAxes, bbox=dict(boxstyle="round",facecolor='none'))

    test_df = test_df.drop(columns=['Data'])
    df_sample = df_sample.drop(columns=['Data'])
    
