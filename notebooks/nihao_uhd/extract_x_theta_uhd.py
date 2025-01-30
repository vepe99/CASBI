import re
import os
import ytree
import glob
from tqdm import tqdm
import pandas as pd
import pynbody as pb
import matplotlib.pyplot as plt 
from multiprocessing import Pool
import numpy as np
import yt



def extract_numeric_part(s):
    return int(s[-5:])


def get_arbor_merge_history(isolated_TreeNode):
    
    df = pd.DataFrame(columns = ['merge_index', 'merge_header', 'merge_id', 'redshift', 'mass_merge_header', 
                                 'mass_progenitor_parallel_to_merge_header', 'file_path'])

    #first we get the main progenitor of the arbor, this is in root -> leaves order
    progenitor_root_to_leaves = list(isolated_TreeNode['prog'])

    #we want the leaves -> root order for all of our data
    progenitor_leaves_to_root = progenitor_root_to_leaves[::-1]
    
    merge_index = 0
    #now we prune the arbor and keep the branch that are not the progenitor of the arbor, we will need the index i
    for i in tqdm(range( len(progenitor_leaves_to_root)) ):
        #if the difference in the number of nodes bewteen two consecutive node in the progenitor_leaves_to_root is bigger than 1 it means that there is another branch:
        l_i = progenitor_leaves_to_root[i].tree_size
        l_i_old = progenitor_leaves_to_root[i-1].tree_size
        if l_i - l_i_old > 1:
            pruned_branches = [j for j in progenitor_leaves_to_root[i]['tree'] if j['uid'] not in progenitor_leaves_to_root[i-1]['tree', 'uid'] ]
            merge_header = [j for j in pruned_branches if j['redshift'] == progenitor_leaves_to_root[i-1]['redshift'] ]
            for m_h in merge_header:
                # if m_h not in list(isolated_TreeNode.get_leaf_nodes()) and len(list(m_h.ancestors)) != 0:
                #     df.loc[len(df)] = [merge_index, m_h, m_h['ID'], m_h['redshift'], m_h['mass'].to('Msun').value.item(), progenitor_leaves_to_root[i-1]['mass'],  list(m_h.ancestors)[0].data_file ]
                # else:  
                df.loc[len(df)] = [merge_index, m_h, m_h['ID'], m_h['redshift'], m_h['M_star'].item(), progenitor_leaves_to_root[i-1]['mass'],  m_h.data_file ]
            merge_index += 1

    return df


def process_row(args, ):
    path, merge_id = args
    sim = pb.load(path)
    # sim.physical_units()
    halos = sim.halos()
    h = halos[merge_id]
    feh = h.s['feh']
    ofe = h.s['ofe']
    infall_time = h.properties['time'].in_units('Gyr')
    mass = h.s['mass'].sum().in_units('Msol')
    return feh, ofe, mass, infall_time

def distribute_operation(df, maximum_merger):
    with Pool(processes=os.cpu_count()) as pool:
        # results = pool.map(process_row, [df.iloc[i] for i in range(maximum_merger)])
        # df_new = df[df['mass_merge_header']<5e10]
        df_new = df
        df_new = df_new.sort_values(by='mass_merge_header', ascending=False)
        list_of_object = [(str(df_new.iloc[i]['file_path'])[:-10], df_new.iloc[i]['merge_id']+1) for i in range(maximum_merger)]
        results = pool.map(process_row, list_of_object)
        
            
    feh_list = [result[0] for result in results]
    ofe_list = [result[1] for result in results]
    mass_list = [result[2] for result in results]
    redsift_list = [result[3] for result in results]
    
    feh_concat = np.concatenate(feh_list)
    ofe_concat = np.concatenate(ofe_list)
    mass_concat = np.array(mass_list)
    infall_time_concat = np.array(redsift_list)
    
    
    return feh_concat, ofe_concat, mass_concat, infall_time_concat

def create_array(i, j):
    final_array = np.zeros((64, 64, 3))
    final_array[:, :, 0 ] =  galaxy_2d_hist
    final_array[:, :, 1] = i*np.ones_like(galaxy_2d_hist)
    final_array[:, :, 2] = j*np.ones_like(galaxy_2d_hist)
    
    return final_array



if __name__ == '__main__':
    for i in [0]:
        path_uhd = '/mnt/storage/_data/nihao/nihao_uhd'
        sim_name = ['8.26e11_zoom_2_new_run/?.??e??.0????', 
                    'g2.79e12_3x6/?.??e??.0????', 
                    'g1.12e12_3x9/?.??e??.0????',
                    'g6.96e11_3x9/?.??e??.0????',
                    'g7.08e11_5x10/?.??e??.0????',
                    'g7.55e11_3x9/?.??e??.0????']
        #we want the last snapshot for each simulations, so the .0???? with the highest value
        sim_path = [sorted(glob.glob(os.path.join(path_uhd, sim)), key=extract_numeric_part)[-1] for sim in sim_name   ]

        path_ytree_files_z0 = [s+'.parameter' for s in sim_path]
        path_to_tree = path_ytree_files_z0[i]
        single_tree = ytree.load(path_to_tree,  hubble_constant=0.67)
        
        if i==0:
            single_tree.add_alias_field('mass','Mhalo',units='Msun/h')
            
        df = get_arbor_merge_history(single_tree[0])
    
        preprocessin_file_training_set = np.load('/mnt/storage/giuseppe_data/data/casbi_rewriting/preprocess_file.npz', allow_pickle=True)
        min_feh, max_feh = np.percentile(preprocessin_file_training_set['FeH'], [0.1, 100])
        min_ofe, max_ofe = np.percentile(preprocessin_file_training_set['OFe'], [0.1, 100])
        
        feh, ofe, mass, infall_time = distribute_operation(df, 100)
        galaxy_2d_hist, _, _ = np.histogram2d(feh, ofe, bins=64, range=[[min_feh, max_feh], [min_ofe, max_ofe]])
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(np.log(galaxy_2d_hist.T), origin='lower', extent=[min_feh, max_feh, min_ofe, max_ofe], aspect='auto', cmap='viridis', )
        ax.set_xlim(min_feh, max_feh)
        ax.set_ylim(min_ofe, max_ofe)
        # Set the font size of the ticks to 20
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        # Add colorbar to the top of each subplot
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', location='right', pad=0)
        cbar.ax.tick_params(labelsize=20)
        fig.savefig(f'./plot/{path_to_tree[-23:-10]}'+'.pdf')
        
        x = np.array([create_array(i, 1) for i in range(100)])
        
        theta = np.zeros((100, 2))
        theta[:, 0] = mass*0.67 #correct for cosmology
        theta[:, 1] = infall_time

        file_name = path_to_tree[-23:-10]
        np.savez(file='./file/'+file_name+'.npz',
                 theta=theta,
                 x=x)