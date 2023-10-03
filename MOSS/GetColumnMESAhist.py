# -*- coding: utf-8 -*-

import numpy as np

# This is a function
def GetColumnMESAhist(filename_history,col):
    """
    
    Parameters:
    -----------
    filename_history : string
        The location of the history.data file
    col : list
        The columns to be read from the history.data file

    Returns:
    --------
    data_cols : list
        The columns from the history.data file

    Notes:
    ------
    This function reads a specified history.data file
    from the MESA output and gives the desired columns
    back.

    Author:     Ylva Götberg
    Date:       14/12 - 2014  
    """
    
    
    # Read the header of the history.data file
    # try:
    header_entries = open(filename_history, "r").readlines()[5]
    header_entries = np.array(header_entries.split())
    index_dict = {item: index for index, item in enumerate(header_entries)}

    # Get the indices based on the order in the smaller list
    indices = [index_dict[item] for item in col]
    #append the missing column indices
    # ncols = len(header_entries)
    # for i in range(ncols):
    #     if i not in indices and header_entries[i] != 'model_number':
    #         indices.append(i)


    # Read the history.data file
    history = np.loadtxt(filename_history, skiprows=6,usecols = indices)
    

    #make record array with column names
    names = list(header_entries[indices])
    #data_cols = np.rec.fromarrays(history.transpose(),names=names)
    data_cols = history.transpose()

    # Create the list which will be filled with data
    # data_cols = list()
    
    # # Iterate through the desired columns    
    # for c in range(0,len(col)):
    #     ind = header_entries.index(col[c])
        
    #     if len(history) == len(header_entries):
    #         data_cols.append(history[ind])
    #     else:
    #         data_cols.append(history[:,ind])
# except:

    #     data_cols = list()        
        
    return data_cols
if __name__=='__main__':
    print("HELLO")
    import os
    from locate_ZAMS import locate_ZAMS
    gnum = '014'
    loc_bin_grid = f'/home/adesai/snap/firefox/common/Downloads/evolutionary_tracks/grid_{gnum}/'
    sims_bin = [name for name in os.listdir(loc_bin_grid) if (name[0]=='M' and os.path.isfile(loc_bin_grid+name+'/LOGS1/history.data'))]
    nbr_sims_bin = len(sims_bin)
    m_bin_grid = [None]*len(sims_bin)
    for i in range(len(sims_bin)):
        m_bin_grid[i] = np.float_(sims_bin[i].split('M1_')[1].split('q')[0])
    ind_sort = np.argsort(np.array(m_bin_grid))
    m_bin_grid = np.array(m_bin_grid)[ind_sort]
    sims_bin = np.array(sims_bin)[ind_sort]
    nbr_sims_bin = len(sims_bin)


    # Storage space
    MESA_params_b = [None]*nbr_sims_bin   # This will be a matrix per model and contain the parameters inside col
    
    # Indices for understanding when the star is halfway during helium burning, detaching, and ends its life
    ind_he = [None]*nbr_sims_bin
    ind_detach = [None]*nbr_sims_bin
    ind_end = [None]*nbr_sims_bin

    # These are the ones I really need:
    mstrip_grid = np.zeros(nbr_sims_bin)
    rstrip_grid = np.zeros(nbr_sims_bin)
    strip_duration_grid = np.zeros(nbr_sims_bin)
    history_filename = 'history.data'
    #'log_surface_h1','log_surface_he4'
    #,'Beq'
    col_bin = ['log_Teff','log_L','center_he4','center_h1','log_Lnuc','he_core_mass','log_R','star_age','log_g','star_mass','surface_n14','surface_c12','surface_o16','surf_avg_v_rot','rl_relative_overflow_1','surface_h1','surface_he4']
    nbr_MESA_param_b = len(col_bin)

    for i in range(nbr_sims_bin):

        # Read the history file
        filename_history = loc_bin_grid+sims_bin[i]+'/LOGS1/'+history_filename
        data = GetColumnMESAhist(filename_history,col_bin)
        
        log_L_b = data['log_L']
        log_Lnuc = data['log_Lnuc']

        # locate the ZAMS
        ind_ZAMS_b = locate_ZAMS(log_L_b, log_Lnuc)   

        # Update the shape of the properties
        # Rewrite the properties to remove the pre-MS
        MESA_params_b[i] = data[:][ind_ZAMS_b:]#[None]*nbr_MESA_param_b
        MESA_params_b[i]['rl_relative_overflow_1'] = MESA_params_b[i]['rl_relative_overflow_1']>0
        #save the record arrays as numpy binary
        np.save(f'/home/adesai/snap/firefox/common/Downloads/evolutionary_tracks/grid_{gnum}_binary_full/{sims_bin[i]}',MESA_params_b[i],allow_pickle=True)
        MESA_params_b[i] = np.load(f'/home/adesai/snap/firefox/common/Downloads/evolutionary_tracks/grid_{gnum}_binary_full/{sims_bin[i]}.npy')
        # for cc in range(nbr_MESA_param_b):
        #     # Store the parameter from ZAMS and onwards
        #     MESA_params_b[i][cc] = data[col_bin[cc]][ind_ZAMS_b:]
        #     # Edit the overflow parameter 
        #     if col_bin[cc] == 'rl_relative_overflow_1':
        #         MESA_params_b[i][cc] = data[col_bin[cc]][ind_ZAMS_b:]>0.

        # Save properties
        indices = np.arange(len(MESA_params_b[i]))
        center_h1_b = MESA_params_b[i]['center_h1']
        surface_h1_b = MESA_params_b[i]['surface_h1']
        center_he4_b = MESA_params_b[i]['center_he4']
        star_mass_b = MESA_params_b[i]['star_mass']
        log_R_b = MESA_params_b[i]['log_R']
        RLOF_b = MESA_params_b[i]['rl_relative_overflow_1']>0
        star_age_b = MESA_params_b[i]['star_age']
        
        # Find halfway through helium burning
        tmp = (center_h1_b < 1e-2)*(center_he4_b > 0.5)
        ind_he[i] = indices[tmp][-1]

        # Record the masses and radii of the stripped stars at this time
        mstrip_grid[i] = star_mass_b[ind_he[i]]
        rstrip_grid[i] = 10**log_R_b[ind_he[i]]

        # Find the duration of the stripped phase
        # I will define that as when the first mass transfer phase stops 
        # until burning no longer is the main reason for the luminosity
        # Mass transfer is not that easy to define for the lowest mass objects, but I will go for the simplification here
        tmp = (center_h1_b < 1e-2)*(surface_h1_b < 0.8*surface_h1_b[0])*(RLOF_b==False)

        ind_detach[i] = indices[tmp][0]  # When detachment occurs

        # Stop before the star is a WD (or has exploded) - this is an ad-hoc solution
        tmp = center_he4_b < 1e-4
        if np.sum(tmp):
            ind_end[i] = np.argmin(np.abs(np.max(log_R_b[tmp])-log_R_b))
        else:
            ind_end[i] = indices[-1]


        # Now, we want to record the duration of the stripped phase, but only if the full stripped phase was finished
        if center_he4_b[-1] < 1e-4:
            strip_duration_grid[i] = star_age_b[ind_end[i]]-star_age_b[ind_detach[i]]
        # print(strip_duration_grid)

        # Tell the log that the model was read
        # fid_log = open(log_name,'a')
        # fid_log.write('At '+sims_bin[i]+'\n')
        # fid_log.close()
        print(i)
