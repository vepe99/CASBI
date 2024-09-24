from scipy.stats import gaussian_kde
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from multiprocessing import cpu_count
from multiprocessing import Pool
import numpy as np

import os
import time
import torch

def pdf(m, m_max, m_min, alpha):
    """
    Power law mass function
    
    Parameters:
    m: mass of the galaxy
    m_max: maximum mass of the galaxy
    m_min: minimum mass of the galaxy
    alpha: power law index of the mass function
    
    Returns:
    pdf: power law mass function value at mass m
    """
    norm_const = (m_max**(1-alpha) - m_min**(1-alpha))/(1-alpha) 
    return (1/norm_const)* m**(-alpha)
def cdf(m, m_max, m_min, alpha):
    """
    Cumulative distribution function of the power law mass function
    
    Parameters:
    m: mass of the galaxy
    m_max: maximum mass of the galaxy
    m_min: minimum mass of the galaxy
    alpha: power law index of the mass function
    
    Returns:
    cdf: cumulative distribution function value at mass m
    """
    
    norm_const = (m_max**(1-alpha) - m_min**(1-alpha))/(1-alpha) 
    return (1/norm_const)* (1/(1-alpha)) * (m**(1-alpha) - m_min**(1-alpha))

def inverse_cdf(y, m_max, m_min, alpha):
    """
    Inverse cumulative distribution function of the power law mass function. It is used to sample analytically the mass function.
    
    Parameters:
    y: random number between 0 and 1
    m_max: maximum mass of the galaxy
    m_min: minimum mass of the galaxy
    alpha: power law index of the mass function
    
    Returns:
    m: mass of the galaxy analytically sampled
    """
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
    M_temp = M_tot
    iteration = 0
    while iteration < 100: #number of max halos to be sampled
        if M_temp < mass_nn.min():
            break
        max_u = cdf(M_temp, m_max, m_min, alpha)
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
                if (((M_tot-M_temp)/(M_tot))<0.95):
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

        M_temp = M_temp - mass_sample
        iteration += 1 
    return samples, masses, times


def gen_real_halo(hist_file_path, j, galaxy_name, M_tot, mass_nn, infall_time, m_max, m_min, galaxies_test=None, d=0.1,  alpha=1.25):
    """
    Generate a real halo by sampling the mass function and then looking for Neighbors in the mass space.
    Returns the histogram of the galaxy, the mass and the infall time of the galaxy.
    If the test set is provided, it checks if the galaxy is already present in the test set, if so it generates a new one untill it is not present anymore.
    
    Parameters:
    hist_file_path: path to the histogram file
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
    arr = np.array([np.load(hist_file_path+f'{s}'+'.npz' )['observables']  for s in samples ])
    #some all the histogram to obtain the 0th channel 
    hist_0 = np.sum( arr, axis=0)
    hist_to_return = [np.stack([np.log1p(hist_0), np.ones_like(hist_0)*i, np.ones_like(hist_0)*j]) for i in range(samples.shape[0])]  #nasty trick to allow to save both the N_th number and the histogram in the same array
    
    masses = np.array(masses)
    infall_time = np.array(infall_time)
    indices = np.argsort(masses)[::-1] #sort the masses in descending order
    
    #reorder masses and infall time
    masses = masses[indices]
    infall_time = infall_time[indices]
    samples = samples[indices]
    
    return hist_to_return, np.column_stack([np.log10(masses), np.log10(infall_time), np.arange(len(masses)), np.ones_like(masses)*j]), np.array([samples for i in range(samples.shape[0])]) # I want for each of the hist to have all the names of the galaxies that contributed to it, I cannot flatten it 


def gen_template_library(hist_file_path, N_sample, galaxy_name, M_tot, mass_nn, infall_time, m_max, m_min, alpha, galaxy_test=None, d=0.1):
    """
    Generate a template library of galaxies by sampling the mass function and then looking for Neighbors in the mass space. 
    It returns the 2d histogram of the galaxies ('observables'), the mass and infall time of the galaxies ('parameters') and the list of subhalos name in the galaxies.
    Both observables and parameters have been tagged with the j-th galaxy index and the i-th subhalo index.
    If the test set is provided, it checks if the galaxy is already present in the test set, if so it generates a new one untill it is not present anymore.
    The observables are 2d histograms with 3 channels, the first channel is the sum of all the subhalos, the second channel is the subhalo index and the third channel is the galaxy index.
    The parameters are the mass, the infall, the subhalo index and the galaxy index.
    
    Parameters:
    hist_file_path: path to the histogram file
    N_sample: number of galaxies to be generated
    galaxy_name: list of galaxy names
    mass_nn: list of galaxy masses
    infall_time: list of galaxy infall times
    m_max: maximum mass of the galaxy
    m_min: minimum mass of the galaxy
    alpha: power law index of the mass function
    galaxy_test: list of set of galaxy names in the test set
    d: percentale of the mass that the sample can be far away from the mass function
    
    Returns:
    flattened_hist_list: 2d histogram of the galaxies
    flattened_param_list: mass, infall time, subhalo index and galaxy index
    galaxies_names: list of set of galaxy names in the test set
    

    """
    with Pool(processes=cpu_count()) as p:
        if galaxy_test is not None:
            #if the test set is provided, we generate the galaxy with a j index that starts from the length of the test set
            result = p.starmap(gen_real_halo, [[hist_file_path, j+len(galaxy_test), galaxy_name, M_tot,  mass_nn, infall_time, m_max, m_min, galaxy_test, d, alpha] for j in range(N_sample)]   )
        else:
            result = p.starmap(gen_real_halo, [[hist_file_path, j, galaxy_name, M_tot, mass_nn, infall_time, m_max, m_min, galaxy_test, d, alpha] for j in range(N_sample)]   )
            
    hist_list, params_list, galaxy_list = zip(*result)

    #create the filter to take only the unique galaxies 
    single_galaxy = [arr[0] for arr in galaxy_list if arr.size > 0]
    unique_indices = list({tuple(arr): i for i, arr in enumerate(map(tuple, single_galaxy))}.values())
    
    #print the number of unique galaxies in the training set
    if galaxy_test is not None:
        print('unique galaxy in the trainig set that are not empty:', len(unique_indices))
    else:
        print('unique galaxy in the test set that are not empty:', len(unique_indices))
    
        
    flattened_hist_list = np.array([item for i, sublist in enumerate(hist_list) if i in unique_indices for item in sublist])
    flattened_param_list = np.array([item for i, sublist in enumerate(params_list) if i in unique_indices for item in sublist])
    galaxies_names = [set(arr[0]) for arr in galaxy_list if arr.size > 0] #list that contains set of names of the galaxy in the test set to compare it with the training set 

    return flattened_hist_list, flattened_param_list, galaxies_names
    
    

def template_input(data, M_tot):
    """
    Filter the data to have only the galaxies with mass smaller than the total mass budget.
    Generate the splitted input for the template library generation.
    
    Parameters:
    data: dataframe with the galaxy data
    M_tot: total mass budget for the galaxy halo
    
    Returns:
    galaxy_name: list of galaxy names
    mass_nn: list of galaxy masses
    infall_time: list of galaxy infall times
    m_max: maximum mass of the galaxy
    m_min: minimum mass of the galaxy
    """
    data['star_log10mass'] = 10**data['star_log10mass']
    data = data[data['star_log10mass']<M_tot]
    mass_name = data[['star_log10mass', 'Galaxy_name', 'infall_time']].drop_duplicates()
    
    #cdf bounderies
    m_min, m_max = mass_name['star_log10mass'].min(), mass_name['star_log10mass'].max()
    
    mass_nn = mass_name['star_log10mass'].values.reshape(-1, 1)
    infall_time = mass_name['infall_time'].values.reshape(-1, 1)
    galaxy_name = mass_name['Galaxy_name'].values.reshape(-1, 1)
    
    return galaxy_name, mass_nn, infall_time, m_max, m_min

    
class TemplateLibrary():
    """
    Template Library class for loading and preprocess the data for the SBI model. 
    The class can be access by the users for inspection of the training and test set by the training_galaxies and test_galaxies attributes, which returns a dictionary with galaxies "observables" and "parameters" and the index of the galaxy in the training and test set.
    The class also has a method get_inference_input() to return the training and test set with the right format to be given to the "CASBI.inference.sbi.train_inference()" function.
    The template library first needs to be instanciated with the path to the galaxy array and the dataframe containing the galaxy data, and it is possible to choose the total mass budject "M_tot", the power-law esponent "alpha" 
    of the luminosity function, the number of bins used to generate the 2D observables histogram and the observational uncertanties stardard deviation "sigma".
    In order to generate training and test set the method gen_libary() needs to be called with the number of galaxies in the test and training set.
    """
    def __init__(self, galaxy_array_path: str, dataframe_path: str, M_tot: float = 1.41e9, alpha = 1.25, d:float = 0.1, bins = 64, sigma: float = 0.0):
        
        self.M_tot = M_tot  
        self.alpha = alpha
        self.d = d
        self.bins = bins
        self.galaxy_array_path = galaxy_array_path
        self.dataframe = pd.read_parquet(dataframe_path)[['star_log10mass', 'Galaxy_name', 'infall_time']].drop_duplicates()
        self.galaxy_names_to_remove = ['g6.31e09.01024', 'g6.31e09.00832', 'g6.31e09.00704', 'g6.31e09.00768', 'g6.31e09.00960', 'g6.31e09.00896']
        self.dataframe = self.dataframe[~self.dataframe['Galaxy_name'].isin(self.galaxy_names_to_remove)]
        self.dataframe['star_log10mass'] = 10**self.dataframe['star_log10mass']
        self.dataframe = self.dataframe[self.dataframe['star_log10mass'] < M_tot]
        
        self.m_min = self.dataframe['star_log10mass'].min()
        self.m_max = self.dataframe['star_log10mass'].max()
        
        self.mass_nn = self.dataframe['star_log10mass'].values.reshape(-1, 1)
        self.infall_time = self.dataframe['infall_time'].values.reshape(-1, 1)
        self.galaxy_name = self.dataframe['Galaxy_name'].values.reshape(-1, 1)
        
        self.sigma = sigma
        
        self.training_galaxies = {}
        self.test_galaxies = {}
    
    def pdf(self, m, m_max, m_min, alpha):
        """
        Power law mass function
        
        Parameters:
        m: mass of the galaxy
        m_max: maximum mass of the galaxy
        m_min: minimum mass of the galaxy
        alpha: power law index of the mass function
        
        Returns:
        pdf: power law mass function value at mass m
        """
        norm_const = (self.m_max**(1-self.alpha) - self.m_min**(1-self.alpha))/(1-self.alpha) 
        return (1/norm_const)* m**(-self.alpha)
    
    def cdf(self, m, m_max, m_min, alpha):
        """
        Cumulative distribution function of the power law mass function
        
        Parameters:
        m: mass of the galaxy
        m_max: maximum mass of the galaxy
        m_min: minimum mass of the galaxy
        alpha: power law index of the mass function
        
        Returns:
        cdf: cumulative distribution function value at mass m
        """
        
        norm_const = (self.m_max**(1-self.alpha) - self.m_min**(1-self.alpha))/(1-self.alpha) 
        return (1/norm_const)* (1/(1-self.alpha)) * (m**(1-self.alpha) - self.m_min**(1-self.alpha))

    def inverse_cdf(self, y, m_max, m_min, alpha):
        """
        Inverse cumulative distribution function of the power law mass function. It is used to sample analytically the mass function.
        
        Parameters:
        y: random number between 0 and 1
        m_max: maximum mass of the galaxy
        m_min: minimum mass of the galaxy
        alpha: power law index of the mass function
        
        Returns:
        m: mass of the galaxy analytically sampled
        """
        norm_const = (self.m_max**(1-self.alpha) - self.m_min**(1-self.alpha))/(1-self.alpha) 
        return (y*norm_const*(1-self.alpha) + self.m_min**(1-self.alpha))**(1/(1-self.alpha))
    
    def gen_subhalo_sample(self, samples, masses, times, nbrs):
        """
        Function to return the Galaxy name, mass and infall time obtain by sampling the mass function and then looking for Neighbors in the mass space.
        If the sample is too far away from the mass function, 5 new samples are drawn and we randomly select one of them, if they are enough close and not already in the sample list.
        If the total mass is not reached we break the loop and return the list of samples, masses and times.
        
        Parameters:
        samples: list of galaxy names
        masses: list of galaxy masses
        times: list of galaxy infall times
        nbrs: fitted nearest neighbors model to the masses         
        """
        M_temp = self.M_tot
        iteration = 0
        while iteration < 100: #naumber of max halos to be sampled
            if M_temp < self.mass_nn.min():
                break
            #check what is the maximum mass that can be sampled from the remaining mass budget and sample it
            max_u = self.cdf(M_temp, self.m_max, self.m_min, self.alpha)
            analictical_sample = self.inverse_cdf(np.random.uniform(0, max_u), self.m_max, self.m_min, self.alpha, ).reshape(-1, 1)
            #find the nearest neighbor to the sampled mass and its indices 
            distances, indices = nbrs.kneighbors(analictical_sample)
            sample = self.galaxy_name[indices[0]][0][0]
            mass_sample = self.mass_nn[indices[0]][0][0]
            time_sample = self.infall_time[indices[0]][0][0]
            #check if the sample is too far away from the mass function or if it is already in the sample list and in case resample 10 new samples
            if (abs(mass_sample - analictical_sample) > self.d*analictical_sample) | (sample in samples):
                analytic_10_samples = self.inverse_cdf(np.random.uniform(0, max_u, size=10), self.m_max, self.m_min, self.alpha, ).reshape(-1, 1)
                distances, indices = nbrs.kneighbors(analytic_10_samples)
                galaxy_10 = self.galaxy_name[indices]
                mass_10 = self.mass_nn[indices]
                time_10 = self.infall_time[indices]
                mask =  (distances < self.d*analytic_10_samples).reshape(galaxy_10.shape)&(~np.isin(galaxy_10, samples))
                if (mask.sum() == 0):
                    if (((self.M_tot-M_temp)/(self.M_tot))<0.95):
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

            M_temp = M_temp - mass_sample
            iteration += 1 
        return samples, masses, times

    def gen_halo(self, j, galaxies_test=None):
        """
        Generate a real halo by sampling the mass function and then looking for Neighbors in the mass space.
        Returns the histogram of the galaxy, the mass and the infall time of the galaxy.
        If the test set is provided, it checks if the galaxy is already present in the test set, if so it generates a new one untill it is not present anymore.
        
        Parameters:
        hist_file_path: path to the histogram file
        j: index of the galaxy to be generated
        galaxies_test: list of galaxy names in the test set
        """
        
        np.random.seed(j + int(time.time()))
        N=2
        nbrs = NearestNeighbors(n_neighbors=N, algorithm='ball_tree').fit(self.mass_nn)
        samples = []
        masses =  []
        times = []
        #generate a milky way galaxy like halo
        samples, masses, times = self.gen_subhalo_sample(samples, masses, times, nbrs)
        
        #check if the milky way like halo is in the test set, otherwise genereate a new one untill is not present anymore in the test set
        if (galaxies_test is not None)&(samples is not None):
            while any(set(samples) == galaxy_in_testset for galaxy_in_testset in galaxies_test):
                samples = []
                masses =  []
                times = []
                samples, masses, times = self.gen_subhalo_sample(samples, masses, times, nbrs)

        if samples is None:
            return np.array([]), np.array([]), np.array([])

        #get the galaxy name 
        samples =  np.array(samples)
        masses = np.array(masses)
        times = np.array(times)
        
        #reorder masses and infall time
        indices = np.argsort(masses)[::-1]
        masses = masses[indices]
        times = times[indices]
        samples = samples[indices]
        
        #path to the galaxy array in the samples
        path = [  os.path.join(self.galaxy_array_path, galaxy_name+'.npz') for galaxy_name in samples]
        
        #load the feh and ofe of the samples and also mass weights to be able to sum the histograms
        feh = np.concatenate( [np.load(path_to_galaxy)['feh'] for path_to_galaxy in path ] )
        ofe = np.concatenate( [np.load(path_to_galaxy)['ofe'] for path_to_galaxy in path ] )
        mass_weight = np.concatenate([ galaxy_mass*np.ones_like(np.load(path_to_galaxy)['feh']) for (galaxy_mass, path_to_galaxy) in zip(masses, path)   ]   ) 
        
        #add Gaussian noise 
        feh = feh + np.random.normal(0, self.sigma, feh.shape)
        ofe = ofe + np.random.normal(0, self.sigma, ofe.shape)
        
        #create the histograms of the samples
        # histogram, _, _ = np.histogram2d(feh, ofe, bins=self.bins, weights=mass_weight)
        histogram, _, _ = np.histogram2d(feh, ofe, bins=self.bins)
        histogram_to_return = [np.stack([np.log10(histogram+1), np.ones_like(histogram)*i, np.ones_like(histogram)*j]) for i in range(samples.shape[0])]  #nasty trick to allow to save both the N_th number and the histogram in the same array
        
        return histogram_to_return, np.column_stack([np.log10(masses), np.log10(times), np.arange(len(masses)), np.ones_like(masses)*j]), np.array([samples for i in range(samples.shape[0])]) # I want for each of the hist to have all the names of the galaxies that contributed to it, I cannot flatten it 

    def gen_libary(self, N_test, N_train):
        """
        Generate the template library of galaxies by sampling the mass function and then looking for Neighbors in the mass space. 
        It instanciate the 2d histogram of the galaxies ('observables'), the mass and infall time of the galaxies ('parameters') as disctionaries with the (i, j) index as keys, j beeing galaxy index and the i-th subhalo index.      
        The parameters are the mass, the infall, the subhalo index and the galaxy index.
        The training and test set are accessible through the training_galaxies, and test_galaxies attributes.
        
        Parameters:
        N_test: number of galaxies in the test set
        N_train: number of galaxies in the training set
        """
        ###TEST SET GENERATION
        with Pool(processes=cpu_count()) as p:
            test_result = p.starmap(self.gen_halo, [[j] for j in range(N_test)])
            p.close()
                      
        hist_list_test, params_list_test, galaxy_list_test = zip(*test_result)

        #create the filter to take only the unique galaxies 
        single_galaxy_test = [arr[0] for arr in galaxy_list_test if arr.size > 0]
        unique_indices_test = list({tuple(arr): i for i, arr in enumerate(map(tuple, single_galaxy_test))}.values())
        
        
        print('unique galaxy in the test set that are not empty:', len(unique_indices_test))
            
        flattened_hist_list_test = np.array([item for i, sublist in enumerate(hist_list_test) if i in unique_indices_test for item in sublist])
        flattened_param_list_test = np.array([item for i, sublist in enumerate(params_list_test) if i in unique_indices_test for item in sublist])
        galaxies_names_test = [set(arr[0]) for arr in galaxy_list_test if arr.size > 0] #list that contains set of names of the galaxy in the test set to compare it with the training set 

        #for easier acess to the test data
        self.x_test = flattened_hist_list_test
        self.params_test = flattened_param_list_test
        
        for k in range(len(flattened_hist_list_test)):
            #add the galaxy to the test set using the (i, j) index
            self.test_galaxies[(flattened_param_list_test[k][2], flattened_param_list_test[k][3])] = {
                'x': flattened_hist_list_test[k][0, :, :], #takes only the true histogram not the index
                'params': flattened_param_list_test[k][:2]
            }

        ###TRAINING SET GENERATION
        with Pool(processes=cpu_count()) as p:
            train_result = p.starmap(self.gen_halo, [[j+len(galaxies_names_test), galaxies_names_test] for j in range(N_train)])
            p.close()
        hist_list_train, params_list_train, galaxy_list_train = zip(*train_result)
        
        single_galaxy_train = [arr[0] for arr in galaxy_list_train if arr.size > 0]
        unique_indices_train = list({tuple(arr): i for i, arr in enumerate(map(tuple, single_galaxy_train))}.values())
        print('unique galaxy in the training set that are not empty:', len(unique_indices_train))
        
        flattened_hist_list_train = np.array([item for i, sublist in enumerate(hist_list_train) if i in unique_indices_train for item in sublist])
        flattened_param_list_train = np.array([item for i, sublist in enumerate(params_list_train) if i in unique_indices_train for item in sublist])
        galaxies_names_train = [set(arr[0]) for arr in galaxy_list_train if arr.size > 0]
        
        #for easier acess to the training data
        self.x_train = flattened_hist_list_train
        self.params_train = flattened_param_list_train
        
        for k in range(len(flattened_hist_list_train)):
            self.training_galaxies[(flattened_param_list_train[k][2], flattened_param_list_train[k][3])] = {
                'x': flattened_hist_list_train[k][0, :, :], #takes only the true histogram not the index
                'params': flattened_param_list_train[k][:2]
            }
        
    def get_inference_input(self):
        """
        Return the training and test set with the right format to be given to the "CASBI.inference.sbi.train_inference()" function.
        """
        self.x_train = torch.tensor(self.x_train)
        self.params_train = torch.tensor(self.params_train)
        self.x_test = torch.tensor(self.x_test)
        self.params_test = torch.tensor(self.params_test)
        return self.x_train, self.params_train, self.x_test, self.params_test
   
    