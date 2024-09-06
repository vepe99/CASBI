from scipy.stats import gaussian_kde
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from multiprocessing import cpu_count
from multiprocessing import Pool
import numpy as np


def pdf(m, m_max, m_min, alpha):
        norm_const = (m_max**(1-alpha) - m_min**(1-alpha))/(1-alpha) 
        return (1/norm_const)* m**(-alpha)
def cdf(m, m_max, m_min, alpha):
    norm_const = (m_max**(1-alpha) - m_min**(1-alpha))/(1-alpha) 
    return (1/norm_const)* (1/(1-alpha)) * (m**(1-alpha) - m_min**(1-alpha))

def inverse_cdf(y, m_max, m_min, alpha):
    norm_const = (m_max**(1-alpha) - m_min**(1-alpha))/(1-alpha) 
    return (y*norm_const*(1-alpha) + m_min**(1-alpha))**(1/(1-alpha))

def gen_non_repeated_halo(samples, masses, times, M_tot, nbrs, d, m_max, m_min, alpha, galaxy_name, mass_nn, infall_time, ):
    """
    Function to return the Galaxy name, mass and infall time obtain by sampling the mass function and then looking for Neighbors in the mass space.
    If the sample is too far away from the mass function, 5 new samples are drawn and we randomly select one of them, if they are enough close and not already in the sample list.
    If the total mass is not reached we break the loop and return the list of samples, masses and times.
    
    Parameters:
    samples: list of galaxy names
    masses: list of galaxy masses
    times: list of galaxy infall times
    M_tot: total mass budject for the galaxy halo
    nbrs: NearestNeighbors object to look for the neighbors in the mass space
    d: percentale of the mass that the sample can be far away from the mass function
    
    Returns:
    samples: updated list of galaxy names
    masses: updated  list of galaxy masses
    times: updated list of galaxy infall times
    
    """
    iteration = 0
    while iteration < 100: #number of max halos to be sampled
        if M_tot < mass_nn.min():
            break
        max_u = cdf(M_tot, m_max, m_min, alpha)
        analictical_sample = inverse_cdf(np.random.uniform(0, max_u), m_max, m_min, alpha, ).reshape(-1, 1)
        distances, indices = nbrs.kneighbors(analictical_sample)
        sample = galaxy_name[indices[0]][0][0]
        mass_sample = mass_nn[indices[0]][0][0]
        time_sample = infall_time[indices[0]][0][0]
        if (abs(mass_sample - analictical_sample) > d*analictical_sample) | (sample in samples):
            nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(mass_nn)
            analytic_10_samples = inverse_cdf(np.random.uniform(0, 1, size=5), m_max, m_min, alpha, ).reshape(-1, 1)
            distances, indices = nbrs.kneighbors(analytic_10_samples)
            galaxy_10 = galaxy_name[indices]
            mass_10 = mass_nn[indices]
            time_10 = infall_time[indices]
            mask =  (distances < d*analytic_10_samples).reshape(galaxy_10.shape)&(~np.isin(galaxy_10, samples))
            if (mask.sum() == 0):
                if (((6*1e9-M_tot)/(6*1e9))<0.95):
                    # print(f'No halos satified the requirement, total mass is: {((6*1e9-M_tot)/(6*1e9))*100:.2f} %') #this is for studying rejection of not completed galaxy
                    samples = None
                    masses = None
                    times = None
                    return samples, masses, times
                else:
                    break #when the 95% of the total mass is reach we keep only those samples and we do not add more
            else:
                sampled_index = np.random.choice(range(len(mass_10[mask].flatten())))
                mass_sample =  mass_10[mask].flatten()[sampled_index]
                sample = galaxy_10[mask].flatten()[sampled_index]
                time_sample  = time_10[mask].flatten()[sampled_index]
        samples.append(sample)
        masses.append(mass_sample)
        times.append(time_sample)

        M_tot = M_tot - mass_sample
        iteration += 1 
    return samples, masses, times


def gen_real_halo(j, galaxy_name, mass_nn, infall_time, m_max, m_min, galaxies_test=None, d=0.1,  alpha=1.25):
    """
    Generate a real halo by sampling the mass function and then looking for Neighbors in the mass space.
    Returns the histogram of the galaxy, the mass and the infall time of the galaxy.
    If the test set is provided, it checks if the galaxy is already present in the test set, if so it generates a new one untill it is not present anymore.
    
    Parameters:
    j: index of the galaxy to be generated
    galaxy_name: list of galaxy names
    mass_nn: list of galaxy masses
    infall_time: list of galaxy infall times
    galaxies_test: list of set of galaxy names in the test set
    d: percentale of the mass that the sample can be far away from the mass function
    
    Returns:
    hist_to_return: histogram of the galaxy, repeated as many time as the number of subhalos
    masses: masses of the subhalos
    infall_time: galaxy name of the subhalos
    """
    
    np.random.seed(j)
    N=2
    nbrs = NearestNeighbors(n_neighbors=N, algorithm='ball_tree').fit(mass_nn)
    M_tot = 6 * 1e9
    samples = []
    masses =  []
    times = []
    #generate a milky way galaxy like halo
    samples, masses, times = gen_non_repeated_halo(samples, masses, times, M_tot, nbrs, d, m_max, m_min, alpha, galaxy_name, mass_nn, infall_time)
    
    #check if the milky way like halo is in the test set, otherwise genereate a new one untill is not present anymore in the test set
    if (galaxies_test is not None)&(samples is not None):
        while any(set(samples) == galaxy_in_testset for galaxy_in_testset in galaxies_test):
            samples = []
            masses =  []
            times = []
            samples, masses, times = gen_non_repeated_halo(samples, masses, times, M_tot, nbrs, d,m_max, m_min, alpha, galaxy_name, mass_nn, infall_time)
    if samples is None:
        return np.array([]), np.array([]), np.array([])

    #get the galaxy name to load the histogram from memory 
    samples =  np.array(samples)
    arr = np.array([np.load('/export/data/vgiusepp/data/full_dataframe/histogram_data/'+f'{s}'+'.npz' )['observables']  for s in samples ])
    #some all the histogram to obtain the 0th channel 
    hist_0 = np.sum( arr, axis=0)
    hist_to_return = [np.stack([hist_0, np.ones_like(hist_0)*i]) for i in range(samples.shape[0])]  #nasty trick to allow to save both the N_th number and the histogram in the same array
    
    masses = np.array(masses)
    infall_time = np.array(infall_time)
    indices = np.argsort(masses)[::-1] #sort the masses in descending order
    
    #reorder masses and infall time
    masses = masses[indices]
    infall_time = infall_time[indices]
    samples = samples[indices]
    
    return hist_to_return, np.column_stack([masses, infall_time]), np.array([samples for i in range(samples.shape[0])]) # I want for each of the hist to have all the names of the galaxies that contributed to it, I cannot flatten it 


def gen_template_library(test_set_sample, galaxy_name, mass_nn, infall_time, m_max, m_min, alpha, galaxy_test=None, d=0.1):
    with Pool(processes=cpu_count()) as p:
        result = p.starmap(gen_real_halo, [[j, galaxy_name, mass_nn, infall_time, m_max, m_min,galaxy_test, d, alpha] for j in range(test_set_sample)]   )
    return result
    