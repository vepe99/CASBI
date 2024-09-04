def save_histograms(galaxy_name, data, min_feh, max_feh, min_ofe, max_ofe ):
    '''
    Save 2d weitghted mass histogram of star masses based on metallicity and alpha abundance for a specific galaxy.
    Parameters:
    - galaxy_name (str): Name of the galaxy.
    - data (DataFrame): DataFrame containing data of all the stars.
    - min_feh (float): Minimum value of metallicity (feh) for the histogram range.
    - max_feh (float): Maximum value of metallicity (feh) for the histogram range.
    - min_ofe (float): Minimum value of alpha abundance (ofe) for the histogram range.
    - max_ofe (float): Maximum value of alpha abundance (ofe) for the histogram range.
    Returns:
    None
    Saves the histograms as a numpy .npz file with the following keys:
    - observables: 2D histogram array of star masses based on metallicity and alpha abundance.
    - star_log10mass: Logarithm of star masses.
    - dm_log10mass: Logarithm of dark matter masses.
    - infall_time: Time of infall.
    - Galaxy_name: Name of the galaxy.
    Example usage:
    save_histograms('Milky Way', data, -2.0, 0.5, -0.5, 0.5)
    '''
    
    galaxy_df = data[data['Galaxy_name'] == galaxy_name]
        
    feh = galaxy_df['feh'].values
    ofe = galaxy_df['ofe'].values
        
    histogram, xedges, yedges = np.histogram2d(feh, ofe, bins=64, range=[[min_feh, max_feh], [min_ofe, max_ofe]], weights=galaxy_df['star_mass'].values)
        
    star_log10mass = galaxy_df['star_log10mass'].values[0]
    dm_log10mass = galaxy_df['dm_log10mass'].values[0]
    infall_time = galaxy_df['infall_time'].values[0]
        
    np.savez(f'../../data/histogram_data_mass/{galaxy_name}.npz', observables=histogram, star_log10mass=star_log10mass, 
             dm_log10mass=dm_log10mass, infall_time=infall_time, Galaxy_name=galaxy_name)
    print(f'save {galaxy_name}')
