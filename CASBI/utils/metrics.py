import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from numpy import random
from scipy.spatial.distance import jensenshannon as js_div
from scipy.stats import kstwobign, pearsonr, gaussian_kde

from tqdm.notebook import tqdm

from CASBI.generator.fff.fff_model import FreeFormFlow
from CASBI.generator.nf.nf_model import NF_condGLOW 

# __all__ = ['ks2d2s', 'estat', 'estat2d']

"""
================================================================================
METRIC TEST
================================================================================
In this package are contained the metric functions for the evaluation of the model.The metric are the Kullback Lieber divergence, the Jensen Shannon divergence and the Kolmogorov Smirov test.
The 2d Kolmogorov Smirnov test (ks2d2s), and all the function linked to it are copied from https://github.com/syrte/ndtest.git.
"""
def ks2d2s(x1, y1, x2, y2, nboot=None, extra=False):
    '''
    Two-dimensional Kolmogorov-Smirnov test on two samples. 
    
    Parameters
    ----------
    x1, y1 : ndarray, shape (n1, )
        Data of sample 1.
    x2, y2 : ndarray, shape (n2, )
        Data of sample 2. Size of two samples can be different.
    nboot : None or int
        Number of bootstrap resample to estimate the p-value. A large number is expected. If None, an approximate analytic estimate will be used.
    extra: bool, optional
        If True, KS statistic is also returned. Default is False.

    Returns
    -------
    p : float
        Two-tailed p-value.
    D : float, optional
        KS statistic, returned if keyword extra is True.

    Notes
    -----
    This is the two-sided K-S test. Small p-values means that the two samples are significantly different. 
    Note that the p-value is only an approximation as the analytic distribution is unkonwn. The approximation is accurate enough when N > ~20 and p-value < ~0.20 or so. When p-value > 0.20, the value may not be accurate,
    but it certainly implies that the two samples are not significantly different. (cf. Press 2007)

    References
    ----------
    Peacock, J.A. 1983, Two-Dimensional Goodness-of-Fit Testing in Astronomy, MNRAS, 202, 615-627
    Fasano, G. and Franceschini, A. 1987, A Multidimensional Version of the Kolmogorov-Smirnov Test, MNRAS, 225, 155-170
    Press, W.H. et al. 2007, Numerical Recipes, section 14.8

    '''
    assert (len(x1) == len(y1)) and (len(x2) == len(y2))
    n1, n2 = len(x1), len(x2)
    D = avgmaxdist(x1, y1, x2, y2)

    if nboot is None:
        sqen = np.sqrt(n1 * n2 / (n1 + n2))
        r1 = pearsonr(x1, y1)[0]
        r2 = pearsonr(x2, y2)[0]
        r = np.sqrt(1 - 0.5 * (r1**2 + r2**2))
        d = D * sqen / (1 + r * (0.25 - 0.75 / sqen))
        p = kstwobign.sf(d)
    else:
        n = n1 + n2
        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, y2])
        d = np.empty(nboot, 'f')
        for i in range(nboot):
            idx = random.choice(n, n, replace=True)
            ix1, ix2 = idx[:n1], idx[n1:]
            #ix1 = random.choice(n, n1, replace=True)
            #ix2 = random.choice(n, n2, replace=True)
            d[i] = avgmaxdist(x[ix1], y[ix1], x[ix2], y[ix2])
        p = np.sum(d > D).astype('f') / nboot
    if extra:
        return p, D
    else:
        return p


def avgmaxdist(x1, y1, x2, y2):
    D1 = maxdist(x1, y1, x2, y2)
    D2 = maxdist(x2, y2, x1, y1)
    return (D1 + D2) / 2


def maxdist(x1, y1, x2, y2):
    n1 = len(x1)
    D1 = np.empty((n1, 4))
    for i in range(n1):
        a1, b1, c1, d1 = quadct(x1[i], y1[i], x1, y1)
        a2, b2, c2, d2 = quadct(x1[i], y1[i], x2, y2)
        D1[i] = [a1 - a2, b1 - b2, c1 - c2, d1 - d2]

    # re-assign the point to maximize difference,
    # the discrepancy is significant for N < ~50
    D1[:, 0] -= 1 / n1

    dmin, dmax = -D1.min(), D1.max() + 1 / n1
    return max(dmin, dmax)


def quadct(x, y, xx, yy):
    n = len(xx)
    ix1, ix2 = xx <= x, yy <= y
    a = np.sum(ix1 & ix2) / n
    b = np.sum(ix1 & ~ix2) / n
    c = np.sum(~ix1 & ix2) / n
    d = 1 - a - b - c
    return a, b, c, d


def kl_divergence(p, q):
 return sum(p[i] * np.log2(p[i]/q[i]) for i in range(len(p)))


def get_test_metric(test_df:pd.DataFrame, df_sample, model:torch.nn.Module):
    """
    Return the KL and JS divergence between the KDE of the true data and the generated data for the test set.
    
    Parameters
    ----------
    test_df : pd.DataFrame
        The test set dataframe.
    
    model: torch.nn.Module
        The model used to generate the data.
    
    Returns
    -------
    kl_div_value : float
        The KL divergence between the KDE of the true data and the generated data.
    js_div_value : float
        The JS divergence between the KDE of the true data and the generated data.
    """
    with torch.no_grad():
        model.eval()
    
        bad_column = ['Galaxy_name']
        other_columns = test_df.columns.difference(bad_column, sort=False)
        
        kl_div_all = np.zeros(len(test_df['Galaxy_name'].unique()))
        js_div_all = np.zeros(len(test_df['Galaxy_name'].unique()))
        D = np.zeros(len(test_df['Galaxy_name'].unique()))
        i = 0
        for galaxy in tqdm(sorted(test_df['Galaxy_name'].unique())):
            galaxy_data = test_df[test_df['Galaxy_name']==galaxy]
            x = galaxy_data['feh']
            y = galaxy_data['ofe']
            kde = gaussian_kde(np.vstack([x, y]))
            kde_value = kde.evaluate(np.vstack([x.to_numpy(), y.to_numpy()]))
            if isinstance(model, NF_condGLOW): 
                flow_pdf  = model.get_pdf(galaxy_data.values[:, :2].astype(float), galaxy_data[other_columns].values[0, 2:].astype(float))
            elif isinstance(model, FreeFormFlow):
                _, flow_logprob = model.log_prob(galaxy_data.values[:, :2].astype(float), galaxy_data[other_columns].values[0, 2:].astype(float))
                flow_pdf = np.exp(flow_logprob.cpu())
            kl_div_all[i] = kl_divergence(kde_value, flow_pdf)
            js_div_all[i]  = js_div(kde_value, flow_pdf)
            
            sample_data = df_sample[df_sample['Galaxy_name']==galaxy]
            x_s = sample_data['feh']
            y_s = sample_data['ofe']
            P, D[i] = ks2d2s(x_s.to_numpy(), y_s.to_numpy(), x.to_numpy(), y.to_numpy(), extra=True)
            i+=1
        
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(131)
        ax.hist(kl_div_all, bins=20)
        ax.set_xlabel('KL divergence')
        ax = fig.add_subplot(132)
        ax.hist(js_div_all, bins=20)
        ax.set_xlabel('JS divergence')
        ax = fig.add_subplot(133)
        ax.hist(D, bins=20)
        ax.set_xlabel('D statistic')
        
        kl_div_mean = np.mean(kl_div_all)
        js_div_mean = np.mean(js_div_all)
        D_mean = np.mean(D)
        return kl_div_mean, js_div_mean, D_mean
    
    
def create_sample_df(df_to_sample, model):
    
    other_cols = df_to_sample.columns.difference(['Galaxy_name'])
    df = pd.DataFrame(columns=df_to_sample.columns)
    with torch.no_grad():
        model.eval()
        for galaxy in sorted(df_to_sample['Galaxy_name'].unique()):    
            galaxy_data = df_to_sample[df_to_sample['Galaxy_name']==galaxy]
            if isinstance(model, NF_condGLOW):
                generated_data = model.sample(len(galaxy_data), galaxy_data[other_cols].values[0,2:]).cpu().detach()
            elif isinstance(model, FreeFormFlow):
                cond = torch.tensor(galaxy_data[other_cols].values[0,2:]).float()
                cond = cond.repeat(len(galaxy_data), 1)
                generated_data = model.sample(len(galaxy_data), cond).cpu().detach()
            generated_data = pd.DataFrame(generated_data.numpy(), columns=['feh', 'ofe'])
            generated_data['Galaxy_name'] = galaxy
            for col in galaxy_data.columns[2:-1]:
                generated_data[col] = galaxy_data[col].values[0]
            df = pd.concat([df, generated_data])
    return df
