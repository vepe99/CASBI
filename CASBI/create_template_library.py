from scipy.stats import gaussian_kde
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from multiprocessing import cpu_count
from multiprocessing import Pool
import numpy as np

import os
import time
import torch

from sklearn.model_selection import train_test_split


    
class TemplateLibrary():
    """
    Template Library class for loading and preprocess the data for the SBI model. 
    The class can be access by the users for inspection of the training and test set by the training_galaxies and test_galaxies attributes, which returns a dictionary with galaxies "observables" and "parameters" and the index of the galaxy in the training and test set.
    The class also has a method get_inference_input() to return the training and test set with the right format to be given to the "CASBI.inference.sbi.train_inference()" function.
    The template library first needs to be instanciated with the path to the galaxy file and the dataframe containing the galaxy data obtained by CASBI.preprocessing, and it is possible to choose the total mass budject "M_tot", the power-law esponent "alpha" 
    of the luminosity function, the number of bins used to generate the 2D observables histogram and the observational uncertanties stardard deviation "sigma".
    In order to generate training and test set the method gen_libary() needs to be called with the number of galaxies in the test and training set.
    
    Parameters:
    
    galaxy_file_path: str
        path to the galaxy files
    
    dataframe_path: str
        path to the dataframe file
    
    
    """
    def __init__(self, 
                 galaxy_file_path: str, 
                 dataframe_path: str, 
                 preprocessing_path: str, 
                 M_tot: float = 1.41e9, 
                 alpha = 1.25, 
                 d:float = 0.1, 
                 bins:int = 64, 
                 sigma: float = 0.0, 
                 perc_feh: float = 0.1,
                 perc_ofe: float = 0.1,
                 galaxy_names_to_remove: list = ['g6.31e09.01024', 'g6.31e09.00832', 'g6.31e09.00704', 'g6.31e09.00768', 'g6.31e09.00960', 'g6.31e09.00896']):
        
        self.M_tot = M_tot  
        self.alpha = alpha
        self.d = d
        self.bins = bins
        self.galaxy_file_path = galaxy_file_path
        self.dataframe = pd.read_parquet(dataframe_path)
        self.galaxy_names_to_remove = galaxy_names_to_remove
        self.dataframe = self.dataframe[~self.dataframe['Galaxy_name'].isin(self.galaxy_names_to_remove)]
        self.dataframe = self.dataframe[self.dataframe['star_mass'] < M_tot]
        
        self.m_min = self.dataframe['star_mass'].min()
        self.m_max = self.dataframe['star_mass'].max()
        
        self.mass_nn = self.dataframe['star_mass'].values.reshape(-1, 1)
        self.infall_time = self.dataframe['infall_time'].values.reshape(-1, 1)
        self.galaxy_name = self.dataframe['Galaxy_name'].values.reshape(-1, 1)
        
        self.sigma = sigma
        
        self.training_galaxies = {}
        self.test_galaxies = {}
        
        self.feh_cut = np.percentile(np.load(preprocessing_path)['FeH'], perc_feh)
        self.ofe_cut = np.percentile(np.load(preprocessing_path)['OFe'], perc_ofe)
        self.feh_lim = [self.feh_cut, self.dataframe['max_feh'].max()]
        self.ofe_lim = [self.ofe_cut, self.dataframe['max_ofe'].max()]
    
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
        # return (m_max**(1-self.alpha) - m**(1-self.alpha)) / (m_min**(1-self.alpha) - m_max**(1-self.alpha))

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
        # """
        norm_const = (self.m_max**(1-self.alpha) - self.m_min**(1-self.alpha))/(1-self.alpha) 
        return (y*norm_const*(1-self.alpha) + self.m_min**(1-self.alpha))**(1/(1-self.alpha))
        # Generate uniform random numbers
        
        # Inverse CDF formula to sample masses
        # samples = (m_min**(1-self.alpha) - y * (m_min**(1-self.alpha) - m_max**(1-self.alpha)))**(1/(1-self.alpha))
        
        # return samples
    
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
        path = [  os.path.join(self.galaxy_file_path, galaxy_name+'.npz') for galaxy_name in samples]
        
        #load the feh and ofe of the samples and also mass weights to be able to sum the histograms
        feh = np.concatenate( [np.load(path_to_galaxy)['feh'][(np.load(path_to_galaxy)['feh']>self.feh_cut)&(np.load(path_to_galaxy)['ofe']>self.ofe_cut)] for path_to_galaxy in path ] )
        ofe = np.concatenate( [np.load(path_to_galaxy)['ofe'][(np.load(path_to_galaxy)['feh']>self.feh_cut)&(np.load(path_to_galaxy)['ofe']>self.ofe_cut)] for path_to_galaxy in path ] )
        mass_weight = np.concatenate([ galaxy_mass*np.ones_like(np.load(path_to_galaxy)['feh'][(np.load(path_to_galaxy)['feh']>self.feh_cut)&(np.load(path_to_galaxy)['ofe']>self.ofe_cut)]) for (galaxy_mass, path_to_galaxy) in zip(masses, path)   ]   ) 
        
        #add Gaussian noise 
        feh = feh + np.random.normal(0, self.sigma, feh.shape)
        ofe = ofe + np.random.normal(0, self.sigma, ofe.shape)
        
        #create the histograms of the samples
        histogram, _, _ = np.histogram2d(feh, ofe, bins=self.bins, range=[self.feh_lim, self.ofe_lim], weights=mass_weight)
        # histogram, _, _ = np.histogram2d(feh, ofe, bins=self.bins)
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
   
    def create_single_halo_library(self, test_percentage):
        """
        Create a template library with a single halo
        """
        #path to the galaxy array in the samples
        # all_galaxy_path = os.listdir(self.galaxy_file_path)
        
        # #let's remove the bad galaxies and all the error
        # words_to_remove = self.galaxy_names_to_remove
        # words_to_remove.append('error')
        # path = [os.path.join(self.galaxy_file_path, galaxy_name) for galaxy_name in all_galaxy_path if not any(word in galaxy_name for word in words_to_remove)]
        
        dwarf_galaxy = self.dataframe['Galaxy_name'][self.dataframe['star_mass']<self.M_tot].values
        path = [os.path.join(self.galaxy_file_path, galaxy_name+'.npz') for galaxy_name in dwarf_galaxy]
        
        #load the feh and ofe of the samples and also mass weights to be able to sum the histograms
        feh = [np.load(path_to_galaxy)['feh'][(np.load(path_to_galaxy)['feh']>self.feh_cut)&(np.load(path_to_galaxy)['ofe']>self.ofe_cut)] for path_to_galaxy in path ]
        ofe = [np.load(path_to_galaxy)['ofe'][(np.load(path_to_galaxy)['feh']>self.feh_cut)&(np.load(path_to_galaxy)['ofe']>self.ofe_cut)] for path_to_galaxy in path ]
        
        
                
        #get the observation
        x = np.array([np.histogram2d(feh[i], ofe[i], bins=self.bins, range=[self.feh_lim, self.ofe_lim])[0] for i in range(len(feh))])
        
        #get the parameters
        param_m  = np.array([np.log10(np.load(path_to_galaxy)['star_mass']) for path_to_galaxy in path])
        param_tau = np.array([np.log10(np.load(path_to_galaxy)['infall_time']) for path_to_galaxy in path])
        params = np.column_stack([param_m, param_tau])
        

        self.x_train, self.x_test, self.params_train, self.params_test = train_test_split(x, params, test_size=test_percentage, random_state=42)
        
        
        
        