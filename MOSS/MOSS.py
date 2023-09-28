#!/usr/bin/env python
# coding: utf-8


"""
#                                                                             #
#                        MOSS, part 1                                         # 
#                                                                             #

#                                                                             #
# version 1.0                                                                 #
# Ylva Gotberg, 27 August 2019                                                #
#                                                                             #
# This is the first version of Models Of Stripped Stars (MOSS) in a regular   #
# python script version. I am now transporting it from ipython notebooks so   #
# that it will be easier to run.                                              #
#                                                                             #
# MOSS was first developed for Gotberg et al. (2019), where we presented the  #
# contribution from stripped stars to the emission from stellar populations.  #
# This is a more advanced version of MOSS where we allow for different        #
# choices of the parameters and it also becomes easier to run.                #
#                                                                             #
# This script contains the population synthesis part of MOSS. It creates a    #
# population taking a Monte Carlo approach and saves the results to           #
# text-files.                                                                 #
#                                                                             #
"""


# Packages needed for running
import numpy as np
import copy
import os
import datetime
from scipy import interpolate
import astropy.units as u
from astropy import constants as const
G = const.G.to('AU3/(M_sun d2)').value   # Gravitational constant in AU^3 / (Msun * day^2)
RSun_AU = (u.Rsun).to(u.AU)        # Solar radius in AU
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
fsize = 20
rc('font',**{'family':'serif','serif':['Times'],'size'   : fsize})
rc('font',**{'size'   : fsize})
rc('text', usetex=True)

# Import some of my own functions
from GetColumnMESAhist import GetColumnMESAhist
from locate_ZAMS import locate_ZAMS
from IMF import *
from GetCMFGENflux import GetCMFGENflux
from GetQs_CMFGEN import GetQs_CMFGEN
from extrapolate import extrapolate
from IBF import *  

# Limits for the code
# This is the limit for the star-formation rate that the memory can handle. If the SFR is higher, we need to take turns 
SFR_lim = 0.0005 #0.005   # MSun/yr



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                                                             #
#     Initial conditions                                                      #
#                                                                             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# Read the input file
filename_input = 'input.txt'
fid_input = open(filename_input,'r')
data_in = fid_input.readlines()
fid_input.close()

# Loop through the lines in the input to get the input 
for i in range(len(data_in)):
    
    if len(data_in[i]) > 2.:
        if '#' in data_in[i].split()[0]: pass
        elif 'run_name' in data_in[i]: run_name = data_in[i].split()[2]
        elif 'duration' in data_in[i]: duration = np.float_(data_in[i].split()[2])
        elif 'Z = ' in data_in[i]: Z = np.float_(data_in[i].split()[2])
        elif 'loc_sin_grids' in data_in[i]: loc_sin_grids = data_in[i].split()[2]
        elif 'loc_bin_grids' in data_in[i]: loc_bin_grids = data_in[i].split()[2]
        elif 'loc_sin_grid' in data_in[i]: loc_sin_grid = data_in[i].split()[2]
        elif 'loc_bin_grid' in data_in[i]: loc_bin_grid = data_in[i].split()[2]
        elif 'loc_pureHe_grid' in data_in[i]: loc_pureHe_grid = data_in[i].split()[2]
        elif 'exclude_pMS' in data_in[i]: exclude_pMS = data_in[i].split()[2] == 'True'
        elif 'type_SF' in data_in[i]: type_SF = data_in[i].split()[2]
        elif 'total_mass_starburst' in data_in[i]: total_mass_starburst = np.float_(data_in[i].split()[2])
        elif 'starformation_rate' in data_in[i]: starformation_rate = np.float_(data_in[i].split()[2]) 
        elif 'evaluation_time' in data_in[i]: 
            evaluation_time = data_in[i].split()[2]
            if ',' in evaluation_time: evaluation_time = np.float_(evaluation_time.split(','))
            else: evaluation_time = np.array([np.float(evaluation_time)])
        elif 'IMF_choice' in data_in[i]: IMF_choice = data_in[i].split()[2]
        elif 'mmin' in data_in[i]: mmin = np.float_(data_in[i].split()[2])
        elif 'mmax' in data_in[i]: mmax = np.float_(data_in[i].split()[2])
        elif 'alpha_IMF' in data_in[i]: alpha_IMF = np.float_(data_in[i].split()[2])
        elif 'IBF_choice' in data_in[i]: IBF_choice = data_in[i].split()[2]
        elif 'Bmin' in data_in[i]: Bmin = np.float_(data_in[i].split()[2])
        elif 'Bmax' in data_in[i]: Bmax = np.float_(data_in[i].split()[2])
        elif 'Bmean' in data_in[i]: Bmean = np.float_(data_in[i].split()[2])
        elif 'Bstdev' in data_in[i]: Bstdev = np.float_(data_in[i].split()[2])
        elif 'Bfrac' in data_in[i]: Bfrac = np.float_(data_in[i].split()[2])
        elif 'Bk' in data_in[i]: Bk = np.float_(data_in[i].split()[2])
        elif 'Bmu' in data_in[i]: Bmu = np.float_(data_in[i].split()[2])
        elif 'loc_B_grids' in data_in[i]: loc_B_grids = data_in[i].split()[2]
        elif 'fbin_choice' in data_in[i]: fbin_choice = data_in[i].split()[2]
        elif 'filename_moe' in data_in[i]: filename_moe = data_in[i].split()[2]
        elif 'fbin_constant' in data_in[i]: fbin_constant = np.float_(data_in[i].split()[2])
        elif 'q_choice' in data_in[i]: q_choice = data_in[i].split()[2]
        elif 'qmin' in data_in[i]: qmin = np.float_(data_in[i].split()[2])
        elif 'qmax' in data_in[i]: qmax = np.float_(data_in[i].split()[2])
        elif 'eta_q' in data_in[i]: eta_q = np.float_(data_in[i].split()[2])
        elif 'P_choice' in data_in[i]: P_choice = data_in[i].split()[2]
        elif 'Mlim_Sana' in data_in[i]: Mlim_Sana = np.float_(data_in[i].split()[2])
        elif 'kappa_P' in data_in[i]: kappa_P = np.float_(data_in[i].split()[2])
        elif 'frac_magnetic' in data_in[i]: frac_magnetic = np.float_(data_in[i].split()[2])
        elif 'q_crit_MS' in data_in[i]: q_crit_MS = np.float_(data_in[i].split()[2])
        elif 'q_crit_HG' in data_in[i]: q_crit_HG = np.float_(data_in[i].split()[2])
        elif 'P_min_crit' in data_in[i]: P_min_crit = np.float_(data_in[i].split()[2])
        elif 'alpha_prescription' in data_in[i]: alpha_prescription = bool(data_in[i].split()[2])
        elif 'alpha_CE' in data_in[i]: alpha_CE = np.float_(data_in[i].split()[2])
        elif 'lambda_CE' in data_in[i]: lambda_CE = np.float_(data_in[i].split()[2])
        elif 'Minit_strip_min' in data_in[i]: Minit_strip_min = np.float_(data_in[i].split()[2])
        elif 'Minit_strip_max' in data_in[i]: Minit_strip_max = np.float_(data_in[i].split()[2])
        elif 'beta_MS' in data_in[i]: beta_MS = np.float_(data_in[i].split()[2])
        elif 'beta_HG_CEE' in data_in[i]: beta_HG_CEE = np.float_(data_in[i].split()[2])
        elif 'beta_HG' in data_in[i]: beta_HG = np.float_(data_in[i].split()[2])
        elif 'angmom' in data_in[i]: angmom = data_in[i].split()[2]
        elif 'gamma_MS' in data_in[i]: gamma_MS = np.float_(data_in[i].split()[2])
        elif 'gamma_HG' in data_in[i]: gamma_HG = np.float_(data_in[i].split()[2])
        elif 'rejuv_choice' in data_in[i]: rejuv_choice = data_in[i].split()[2]
        elif 'record_stars' in data_in[i]: record_stars = data_in[i].split()[2]
        elif 'minimum_mass_to_print' in data_in[i]: minimum_mass_to_print = np.float_(data_in[i].split()[2])
        elif 'maximum_mass_to_print' in data_in[i]: maximum_mass_to_print = np.float_(data_in[i].split()[2])
        elif 'save_figs' in data_in[i]: save_figs = data_in[i].split()[2] == 'True'
        elif 'col' in data_in[i]: col = data_in[i].split()[2].split(',')
        elif 'history_filename' in data_in[i]: history_filename = data_in[i].split()[2]
            


# Check if binaries should be included
compute_binaries = True
if fbin_choice == 'constant':
    if fbin_constant == 0.0: 
        compute_binaries = False
        
# --- UPDATE HERE ---
nbr_MESA_param = len(col)
if compute_binaries:
    col_bin = copy.copy(col)
    col_bin.append('rl_relative_overflow_1')
    nbr_MESA_param_b = len(col_bin)


# This is how the single star model history files are identified
sims = [name for name in os.listdir(loc_sin_grid) if (os.path.isfile(loc_sin_grid+name+'/LOGS/'+history_filename))]
nbr_sims = len(sims)
# My naming has also worked for mass sorting, but this will need to be updated too
m_grid = [None]*nbr_sims
for i in range(nbr_sims):
    #m_grid[i] = np.float_(sims[i].split('M')[1])
    m_grid[i] = np.float_(sims[i])
# Sort them with mass
ind_sort = np.argsort(np.array(m_grid))
m_grid = np.array(m_grid)[ind_sort]
sims = np.array(sims)[ind_sort]
    
# Necessary properties: log_L & log_Lnuc for ZAMS determination, center_h1 for TAMS finding, center_he4 for HG location, log_R for maximum radius during evolutionary stages, log_L and log_Teff for Hayashi track, star_age for lifetime
# --- --- --- ---
    

# This is just to make sure the code is running and the file-saving is working
if save_figs:
    # Location of test-plots
    plots_dir = 'verification_plots'
    os.system('mkdir '+plots_dir)
    
    """
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    ax.plot(np.arange(10),np.arange(10),'.')
    fig.savefig(plots_dir+'/test.png',format='png',bbox_inches='tight',pad_inches=0.1)
    plt.close(fig)
    """


# Initiate a log-file that tracks the progress of the code
# Name of the log-file
log_name = 'log_'+run_name+'.log'

# Also, tell the log the time and that the input file was read
fid_log = open(log_name,'a')
fid_log.write('\n       This is the log for '+run_name+'\n\n')
fid_log.write('Now the date and time is '+str(datetime.datetime.now())+'\n')
fid_log.write('The input file was read.\n\n')

fid_log.write(' ----- SUMMARY OF INPUT ----- \n\n')

if compute_binaries:
    fid_log.write('Binary stars are included\n')
else:
    fid_log.write('Only single stars are computed - no binaries!!!\n')

# Assigning automatic grid locations according to the metallicity if 
# the specific grid is not assigned in the input file. 

# Single star grid
if 'loc_sin_grid' in locals():
    fid_log.write('The location of the single star evolutionary model grid is '+loc_sin_grid+'\n')
else:
    fid_log.write('Assuming automatic grid location for single stars\n')
    loc_sin_grid = loc_sin_grids+'grid_'+str(Z).split('.')[1]+'/'
    fid_log.write('The location of the single star evolutionary model grid is '+loc_sin_grid+'\n')

# Binary star grid
if 'loc_bin_grid' in locals():
    fid_log.write('The location of the binary star evolutionary model grid is '+loc_bin_grid+'\n')
elif 'loc_bin_grids' in locals():
    fid_log.write('Assuming automatic grid location for binary stars\n')
    loc_bin_grid = loc_bin_grids+'grid_'+str(Z).split('.')[1]+'/'
    fid_log.write('The location of the binary star evolutionary model grid is '+loc_bin_grid+'\n')
fid_log.write('\n')
    
# Magnetic star grid
compute_magnetic = False
if 'loc_B_grids' in locals():
    compute_magnetic = True
    fid_log.write('The location of the magnetic star grid is '+loc_B_grids+'\n')
    # Check what grids there are
    B_grids = [name for name in os.listdir(loc_B_grids) if os.path.isdir(loc_B_grids+name+'/5.0')]
    # Sort them according to B-field strength
    B_strength_grids = np.float_(np.array(B_grids))
    ind_sort = np.argsort(B_strength_grids)
    B_grids = np.array(B_grids)[ind_sort]
    B_strength_grids = B_strength_grids[ind_sort]
    nbr_B_grids = len(B_grids)
    # Get the models within the grids 
    B_sims = [None]*nbr_B_grids
    nbr_B_sims = [None]*nbr_B_grids
    B_m_grid = [None]*nbr_B_grids
    for j in range(nbr_B_grids):
        # Tell the log what B-field grids there are 
        fid_log.write('There is a grid with initial magnetic field strengths of '+str(B_strength_grids[j])+' G\n')
        B_sims[j] = [name for name in os.listdir(loc_B_grids+B_grids[j]) if os.path.isfile(loc_B_grids+B_grids[j]+'/'+name+'/LOGS/'+history_filename)]
        B_m_grid[j] = np.float_(np.array(B_sims[j]))
        # Sort them according to mass
        ind_sort = np.argsort(B_m_grid[j])
        B_m_grid[j] = B_m_grid[j][ind_sort]
        B_sims[j] = np.array(B_sims[j])[ind_sort]
        nbr_B_sims[j] = len(B_sims[j])

    
if 'exclude_pMS' in locals():
    fid_log.write('Excluding post-MS evolution, the star dies at TAMS...\n\n')
    
# Repeat the input parameters in the log-file, just for the record. 
# Metallicity
fid_log.write('The metallicity is Z = '+str(Z)+'\n\n')
# Starburst or continuous star-formation
fid_log.write('This is using '+type_SF+' starformation type\n')
if type_SF == 'constant':
    fid_log.write('The star-formation rate is '+str(starformation_rate)+' Msun/yr \n')
elif type_SF == 'starburst':
    fid_log.write('Going to model a starburst with '+str(total_mass_starburst)+' MSun\n')
fid_log.write('\n')
# Initial mass function
fid_log.write('Using an IMF from '+IMF_choice+'\n')
fid_log.write('Setting the mass limits to '+str(mmin)+'-'+str(mmax)+'MSun\n\n')
# Binary fraction
fid_log.write('Using the binary fraction setting: '+fbin_choice+'\n')
if fbin_choice == 'constant':
    fid_log.write('The fraction of stars born in binaries is set to: '+str(fbin_constant)+'\n')
fid_log.write('\n')
        
# This is for the binary part of MOSS
if compute_binaries:
    # Mass ratio distribution
    fid_log.write('The mass ratio distribution is: '+q_choice+'\n')
    fid_log.write('We allow for mass ratios between '+str(qmin)+' < M2/M1 < '+str(qmax)+'\n\n')
    # Period distribution
    fid_log.write('The period distribution is: '+P_choice+'\n')
    if P_choice == 'Opik_Sana':
        fid_log.write('The switch from Opik to Sana occurs at '+str(Mlim_Sana)+' MSun\n')
        fid_log.write('The minimum period for the Sana+12 part is '+str(10**0.15)+'days, while the Opik part goes to ZAMS radius.\n')
    else:
        fid_log.write('The minimum period is set to '+str(P_min_crit)+' days\n')
    if fbin_choice == 'moe':
        P_max = 10**3.7
    else:
        P_max = 10**3.5
    fid_log.write('The maximum period is set to '+str(P_max)+'days\n\n')
    # Critical mass ratios
    fid_log.write('Common envelope on main sequence develops if the mass ratio is <'+str(q_crit_MS)+'\n')
    fid_log.write('Common envelope on Hertzsprung gap develops if the mass ratio is <'+str(q_crit_HG)+'\n\n')
    # Common envelope evolution
    if alpha_prescription:
        fid_log.write('The alpha-prescription is used for modeling common envelopes (initiated on HG, otherwise assumed merge)\n')
        fid_log.write('The efficiency factor, alpha, for common envelope is set to: '+str(alpha_CE)+'\n')
        fid_log.write('The structure parameter, lambda, for common envelope is set to: '+str(lambda_CE)+'\n')
    else:
        fid_log.write('No alpha-prescription: simply assume that common envelopes initiated on the Hertzsprung gap leads to the creation of a stripped star if the mass ratio was larger than the critical value and the Roche-lobe was filled when the envelope of the donor was convective.\n')
    fid_log.write('\n')
    # Limits for the stripped star formation 
    fid_log.write('We assume that stripped stars can be formed from stars with Minit ='+str(Minit_strip_min)+'-'+str(Minit_strip_max)+'MSun\n')
    fid_log.write('This mass range is actually the only mass range we consider binary interaction active\n\n')
    # Mass accretion efficiency
    fid_log.write('The mass accretion efficiency on the MS (Case A) is set to: '+str(beta_MS)+'\n')
    fid_log.write('The mass accretion efficiency on the HG (Case B) is set to: '+str(beta_HG)+'\n')
    fid_log.write('The mass accretion during common envelope initiated on the HG (Case B_CEE) is set to: '+str(beta_HG_CEE)+'\n\n')
    # Treatment of angular momentum        
    fid_log.write('Angular momentum during mass transfer is treated as: '+angmom+'\n')
    if angmom == 'gamma_constant':
        fid_log.write('The parameter gamma is set to constant\n')
        fid_log.write('On the main sequence, gamma = '+str(gamma_MS)+'\n')
        fid_log.write('On the Hertzsprung gap, gamma = '+str(gamma_HG)+'\n')
    fid_log.write('\n')
    fid_log.write('Rejuvenation is set as: '+rejuv_choice+'\n')
fid_log.write('\n  ------------------- \n')
    
# Close the log-file
fid_log.close()

    
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                                                             #
#     BASE: Evolutionary models                                               #
#           - Single stars                                                    #
#                                                                             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 
# Let's start by reading information from the stellar evolutionary models made 
# for single stars. Using these, I can determine when mass transfer will occur.
#
# Read the single star evolutionary models to get:
#   - the radius limits
#   - the lifetime of a single star
#   - the main sequence duration

if compute_magnetic == False:
    # Tell the log what is going to happen
    fid_log = open(log_name,'a')
    fid_log.write('Going to read the evolutionary models for single stars...\n')
    fid_log.close()

    # Storage space
    MESA_params = [None]*nbr_sims   # This will be a matrix per model and contain the parameters inside col

    # Indices for various evolutionary stages
    ind_ZAMS = [None]*nbr_sims
    ind_TAMS = [None]*nbr_sims
    ind_maxR_MS = [None]*nbr_sims  # the index for the maximum radius during main sequence evolution
    ind_maxR_HG = [None]*nbr_sims  # the index for the maximum radius during the Hertzsprung gap
    ind_conv_env = [None]*nbr_sims  # approximation for the time the star reaches the Hayashi track 
                                    # and develops a deep convective envelope


    # These are the parameters I really need
    R_ZAMS_grid = np.zeros(nbr_sims)
    R_maxMS_grid = np.zeros(nbr_sims)
    R_conv_grid = np.zeros(nbr_sims)
    R_maxHG_grid = np.zeros(nbr_sims)

    lifetime_grid = np.zeros(nbr_sims)
    MS_duration_grid = np.zeros(nbr_sims)
    
    # The helium core masses at TAMS to use for masses of stripped stars more massive 
    # than what we have in the binary grid
    he_core_mass_TAMS = np.zeros(nbr_sims)


    # Loop over the models
    for i in range(nbr_sims):

        # --- UPDATE HERE ---
        # -> need better storaging and maybe also reading of mesa files
        # Read the history file
        filename_history = loc_sin_grid+sims[i]+'/LOGS/'+history_filename
        data = GetColumnMESAhist(filename_history,col)

        # Get some of the properties
        log_L = data[col.index('log_L')]
        log_Lnuc = data[col.index('log_Lnuc')]

        # locate the ZAMS
        ind_ZAMS[i] = locate_ZAMS(log_L, log_Lnuc)

        # Rewrite the properties to remove the pre-MS
        MESA_params[i] = [None]*nbr_MESA_param
        for cc in range(nbr_MESA_param):
            # Store the parameter from ZAMS and onwards
            MESA_params[i][cc] = data[cc][ind_ZAMS[i]:]
            # If the parameter is the age, put ZAMS at zero
            if col[cc] == 'star_age':
                MESA_params[i][cc] = MESA_params[i][cc]-MESA_params[i][cc][0]        

        # Re-set the ZAMS index to zero
        ind_ZAMS[i] = 0

        # Some parameters needed below
        log_R = MESA_params[i][col.index('log_R')]
        log_L = MESA_params[i][col.index('log_L')]
        log_Teff = MESA_params[i][col.index('log_Teff')]
        center_h1 = MESA_params[i][col.index('center_h1')]
        center_he4 = MESA_params[i][col.index('center_he4')]
        he_core_mass = MESA_params[i][col.index('he_core_mass')]
        star_age = MESA_params[i][col.index('star_age')]

        # Find the TAMS
        indices = np.arange(len(MESA_params[i][0]))
        ind_TAMS[i] = indices[center_h1 < 1e-2][0]

        # Find the maximum radius during the main sequence
        ind_MS = indices[ind_ZAMS[i]:ind_TAMS[i]]
        ind_maxR_MS[i] = indices[np.max(log_R[ind_MS]) == log_R]

        # Find the maximum radius during the Hertzsprung gap evolution
        ind_HG = (center_h1 < 1e-2)*(center_he4 > 0.98)
        ind_maxR_HG[i] = indices[ind_HG][log_R[ind_HG] == np.max(log_R[ind_HG])]

        # Approximation for the development of the convective envelope (Hayashi track)    
        ind2 = (log_Teff<4.)*(indices > ind_TAMS[i])
        if np.sum(ind2) != 0:
            ind_L = np.argmin(np.abs(np.min(log_L[ind2]) - log_L[ind2]))
            ind_conv_env[i] = indices[ind2][ind_L]
        else:
            ind_conv_env[i] = copy.copy(ind_maxR_HG[i])

        # The radius limits
        R_ZAMS_grid[i] = 10**log_R[ind_ZAMS[i]]
        R_maxMS_grid[i] = 10**log_R[ind_maxR_MS[i]]
        R_conv_grid[i] = 10**log_R[ind_conv_env[i]]
        R_maxHG_grid[i] = 10**log_R[ind_maxR_HG[i]]

        # Record also the lifetimes
        if (center_he4[-1] < 1e-3):
            lifetime_grid[i] = star_age[-1]
        # # # # TEMPORARY # # # # 
        # Temporary for Zsolt's grid
        else:
            lifetime_grid[i] = star_age[-1]
        # # # # # # # # # 

        # Duration of the main sequence evolution
        MS_duration_grid[i] = star_age[ind_TAMS[i]]
        
        # We want also the helium core masses to use for stripped stars that 
        # are more massive than what we have in the binary grid - use TAMS mass
        #he_core_mass_TAMS[i] = he_core_mass[ind_TAMS[i]]   # - this is not working well - let's be more precise
        # Increase a little bit the time when we pick He core mass to not get weird values
        #ii = np.argmin(np.abs(star_age-(1.05*star_age[ind_TAMS[i]])))
        #he_core_mass_TAMS[i] = he_core_mass[ii]
        # This is what inside Beryl's plot - produces the smoothest mass relations
        ind_tmp = center_h1 < 1e-5
        if np.sum(ind_tmp) == 0: 
            ind_hetmp = np.arange(len(center_h1))[-1]
        else:
            ind_hetmp = np.arange(len(center_h1))[ind_tmp][0]      

        he_core_mass_TAMS[i]=he_core_mass[ind_hetmp]
 
        # Tell the log that you have read the model
        fid_log = open(log_name,'a')
        fid_log.write('At '+sims[i]+'\n')
        fid_log.close()


    # Tell the log that the evolutionary models for single stars have been read
    fid_log = open(log_name,'a')
    fid_log.write('The evolutionary models for single stars have been read. \n')
    fid_log.close()

    # Move out the index saying whether a star in the grid lives (past central He exhaustion)
    ind_life = lifetime_grid > 0.

    # Test-figures
    if save_figs:
        # # # # Plot the HRD # # # # # # # #
        fig, ax = plt.subplots(1,1,figsize=(8,6))
        for i in range(nbr_sims):
            log_Teff = MESA_params[i][col.index('log_Teff')]
            log_L = MESA_params[i][col.index('log_L')]
            ax.plot(log_Teff,log_L,'-k')
            if i == 0:
                ax.plot(log_Teff[ind_ZAMS[i]],log_L[ind_ZAMS[i]],'or',label='ZAMS')
                ax.plot(log_Teff[ind_TAMS[i]],log_L[ind_TAMS[i]],'ob',label='TAMS')
                ax.plot(log_Teff[ind_maxR_MS[i]],log_L[ind_maxR_MS[i]],'og',label='max R during MS')
                ax.plot(log_Teff[ind_conv_env[i]],log_L[ind_conv_env[i]],'oy',label='Convective envelope starts')
                ax.plot(log_Teff[ind_maxR_HG[i]],log_L[ind_maxR_HG[i]],'om',label='End HG')
            else:
                ax.plot(log_Teff[ind_ZAMS[i]],log_L[ind_ZAMS[i]],'or')
                ax.plot(log_Teff[ind_TAMS[i]],log_L[ind_TAMS[i]],'ob')
                ax.plot(log_Teff[ind_maxR_MS[i]],log_L[ind_maxR_MS[i]],'og')
                ax.plot(log_Teff[ind_conv_env[i]],log_L[ind_conv_env[i]],'oy')
                ax.plot(log_Teff[ind_maxR_HG[i]],log_L[ind_maxR_HG[i]],'om')
        ax.set_xlabel('$\\log_{10} (T_{\\mathrm{eff}}/\\mathrm{K})$')
        ax.set_ylabel('$\\log_{10} (L/L_{\\odot})$')
        ax.invert_xaxis()
        ax.legend(loc=0,fontsize=0.8*fsize)
        ax.set_title('$Z = '+str(Z)+'$')
        fig.savefig(plots_dir+'/HRD_single.png',format='png',bbox_inches='tight',pad_inches=0.1)
        plt.close(fig)
        # # # # # # # # # # # # # # # # # # 

        # # # # Plot the radius evolution # # # # # # # #
        fig, ax = plt.subplots(1,1,figsize=(8,6))
        xlim = [1e6,2e9]
        for i in range(nbr_sims):
            star_age = MESA_params[i][col.index('star_age')]
            log_R = MESA_params[i][col.index('log_R')]
            ax.plot(star_age,10**log_R,'k-')
            if  i == 0:
                ax.plot(star_age[ind_maxR_MS[i]],10**log_R[ind_maxR_MS[i]],'go',label='max R during MS')
                ax.plot(star_age[ind_conv_env[i]],10**log_R[ind_conv_env[i]],'yo',label='Convective envelope starts')
                ax.plot(star_age[ind_maxR_HG[i]],10**log_R[ind_maxR_HG[i]],'mo',label='End HG')
            else:
                ax.plot(star_age[ind_maxR_MS[i]],10**log_R[ind_maxR_MS[i]],'go')
                ax.plot(star_age[ind_conv_env[i]],10**log_R[ind_conv_env[i]],'yo')
                ax.plot(star_age[ind_maxR_HG[i]],10**log_R[ind_maxR_HG[i]],'mo')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(xlim)
        ax.set_xlabel('Time [yrs]')
        ax.set_ylabel('Radius [$R_{\\odot}$]')
        ax.legend(loc=0,fontsize=0.8*fsize)
        fig.savefig(plots_dir+'/Revol_single.png',format='png',bbox_inches='tight',pad_inches=0.1)
        plt.close(fig)
        # # # # # # # # # # # # # # # # # #

        # # # # Plot the lifetime # # # # # # # #
        fig, ax = plt.subplots(1,1,figsize=(8,6))
        ax.loglog(m_grid[ind_life],lifetime_grid[ind_life],'ko',label='lifetime')
        ax.loglog(m_grid,MS_duration_grid,'ro',label='MS duration')
        ax.set_xlabel('Initial mass [$M_{\\odot}$]')
        ax.set_ylabel('Time [yrs]')
        ax.legend(loc=0,fontsize=0.7*fsize)
        fig.savefig(plots_dir+'/lifetime_single.png',format='png',bbox_inches='tight',pad_inches=0.1)
        plt.close(fig)
        # # # # # # # # # # # # # # # # # #

        # Tell the log 
        fid_log = open(log_name,'a')
        fid_log.write('Saved an HRD, radius evolution, and lifetime diagrams in '+plots_dir+'... \n\n')
        fid_log.close()

    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                                                             #
#     BASE: Evolutionary models                                               #
#           - Magnetic stars                                                  #
#                                                                             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 
# Need to read the evolutionary models for magnetic stars if these are going to 
# be used. Check that first. 
#
# Read the magnetic single star evolutionary models to get:
#   - the radius limits
#   - the lifetime of a single star
#   - the main sequence duration

# First, check if this is going to be used
if compute_magnetic: 
    
    # Tell the log
    fid_log = open(log_name,'a')
    fid_log.write('Going to read the evolutionary models for the magnetic stars...\n')
    fid_log.close()
    
    # Make some storage space
    ind_ZAMS_B = [None]*nbr_B_grids
    ind_TAMS_B = [None]*nbr_B_grids
    ind_maxR_MS_B = [None]*nbr_B_grids
    ind_maxR_HG_B = [None]*nbr_B_grids
    ind_conv_env_B = [None]*nbr_B_grids
    
    MESA_params_B = [None]*nbr_B_grids
    
    R_ZAMS_grid_B = [None]*nbr_B_grids
    R_maxMS_grid_B = [None]*nbr_B_grids
    R_conv_grid_B = [None]*nbr_B_grids
    R_maxHG_grid_B = [None]*nbr_B_grids
    
    lifetime_grid_B = [None]*nbr_B_grids
    MS_duration_grid_B = [None]*nbr_B_grids
    
    
    # Go through the magnetic grids and read the models 
    for j in range(nbr_B_grids): 
        
        # Tell the log
        fid_log = open(log_name,'a')
        fid_log.write('   Inside B-grid '+B_grids[j]+'...\n')
        fid_log.close()        
        
        # Make some storage space
        ind_ZAMS_B[j] = [None]*nbr_B_sims[j]
        ind_TAMS_B[j] = [None]*nbr_B_sims[j]
        ind_maxR_MS_B[j] = [None]*nbr_B_sims[j]
        ind_maxR_HG_B[j] = [None]*nbr_B_sims[j]
        ind_conv_env_B[j] = [None]*nbr_B_sims[j]
        
        MESA_params_B[j] = [None]*nbr_B_sims[j]
        
        R_ZAMS_grid_B[j] = [None]*nbr_B_sims[j]
        R_maxMS_grid_B[j] = [None]*nbr_B_sims[j]
        R_conv_grid_B[j] = [None]*nbr_B_sims[j]
        R_maxHG_grid_B[j] = [None]*nbr_B_sims[j]
        
        lifetime_grid_B[j] = [None]*nbr_B_sims[j]
        MS_duration_grid_B[j] = [None]*nbr_B_sims[j]
        
        
        # Read the models for each B-grid
        for i in range(nbr_B_sims[j]): 
            
            # Read the history file
            filename_history = loc_B_grids+B_grids[j]+'/'+B_sims[j][i]+'/LOGS/'+history_filename
            data = GetColumnMESAhist(filename_history,col)

            # Get some of the properties
            log_L = data[col.index('log_L')]
            log_Lnuc = data[col.index('log_Lnuc')]

            # locate the ZAMS
            ind_ZAMS_B[j][i] = locate_ZAMS(log_L, log_Lnuc)

            # Rewrite the properties to remove the pre-MS
            MESA_params_B[j][i] = [None]*nbr_MESA_param
            for cc in range(nbr_MESA_param):
                # Store the parameter from ZAMS and onwards
                MESA_params_B[j][i][cc] = data[cc][ind_ZAMS_B[j][i]:]
                # If the parameter is the age, put ZAMS at zero
                if col[cc] == 'star_age':
                    MESA_params_B[j][i][cc] = MESA_params_B[j][i][cc]-MESA_params_B[j][i][cc][0]        

            # Re-set the ZAMS index to zero
            ind_ZAMS_B[j][i] = 0

            # Some parameters needed below (temporary, used only in this loop)
            log_R = MESA_params_B[j][i][col.index('log_R')]
            log_L = MESA_params_B[j][i][col.index('log_L')]
            log_Teff = MESA_params_B[j][i][col.index('log_Teff')]
            center_h1 = MESA_params_B[j][i][col.index('center_h1')]
            center_he4 = MESA_params_B[j][i][col.index('center_he4')]
            star_age = MESA_params_B[j][i][col.index('star_age')]

            # Find the TAMS
            indices = np.arange(len(MESA_params_B[j][i][0]))
            ind_TAMS_B[j][i] = indices[center_h1 < 1e-2][0]

            # Find the maximum radius during the main sequence
            ind_MS = indices[ind_ZAMS_B[j][i]:ind_TAMS_B[j][i]]
            ind_maxR_MS_B[j][i] = indices[np.max(log_R[ind_MS]) == log_R]

            # Find the maximum radius during the Hertzsprung gap evolution
            ind_HG = (center_h1 < 1e-2)*(center_he4 > 0.98)
            if np.sum(ind_HG):
                ind_maxR_HG_B[j][i] = indices[ind_HG][log_R[ind_HG] == np.max(log_R[ind_HG])]
            else:
                ind_maxR_HG_B[j][i] = indices[-1]

            # Approximation for the development of the convective envelope (Hayashi track)    
            ind2 = (log_Teff<4.)*(indices > ind_TAMS_B[j][i])
            if np.sum(ind2) != 0:
                ind_L = np.argmin(np.abs(np.min(log_L[ind2]) - log_L[ind2]))
                ind_conv_env_B[j][i] = indices[ind2][ind_L]
            else:
                ind_conv_env_B[j][i] = copy.copy(ind_maxR_HG_B[j][i])

            # The radius limits
            R_ZAMS_grid_B[j][i] = 10**log_R[ind_ZAMS_B[j][i]]
            R_maxMS_grid_B[j][i] = 10**log_R[ind_maxR_MS_B[j][i]]
            R_conv_grid_B[j][i] = 10**log_R[ind_conv_env_B[j][i]]
            R_maxHG_grid_B[j][i] = 10**log_R[ind_maxR_HG_B[j][i]]

            # Record also the lifetimes
            if (center_he4[-1] < 1e-3):
                lifetime_grid_B[j][i] = star_age[-1]
            # # # # TEMPORARY # # # # 
            # Temporary for Zsolt's grid
            else:
                lifetime_grid_B[j][i] = star_age[-1]
            # # # # # # # # # 

            # Duration of the main sequence evolution
            MS_duration_grid_B[j][i] = star_age[ind_TAMS_B[j][i]]

            # Tell the log that you have read the model
            fid_log = open(log_name,'a')
            fid_log.write('At '+B_sims[j][i]+'\n')
            fid_log.close()


    # Tell the log that the evolutionary models for magnetic stars have been read
    fid_log = open(log_name,'a')
    fid_log.write('The evolutionary models for magnetic stars have been read. \n')
    fid_log.close()
            
    


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                                                             #
#     BASE: Evolutionary models                                               #
#           - Stripped stars (binary star models)                             #
#                                                                             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 
# Read the binary star evolutionary models (stripped stars) to get:
#   - the radius, picked midway (central helium mass fraction = 0.5)
#   - the mass, picked midway
#   - the duration of the stripped phase

# This is only done if binaries are included in the simulation
if compute_binaries:

    # Tell the log that the stripped star evolutionary models will be read
    fid_log = open(log_name,'a')
    fid_log.write('Going to read the evolutionary models for stars stripped in binaries... \n')
    fid_log.close()


    # Which models are available?
    sims_bin = [name for name in os.listdir(loc_bin_grid) if (name[0]=='M' and os.path.isfile(loc_bin_grid+name+'/LOGS1/history.data'))]
    m_bin_grid = [None]*len(sims_bin)
    for i in range(len(sims_bin)):
        m_bin_grid[i] = np.float_(sims_bin[i].split('M1_')[1].split('q')[0])
    # Sort them in mass range
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

    # Loop over the evolutionary models
    for i in range(nbr_sims_bin):

        # Read the history file
        filename_history = loc_bin_grid+sims_bin[i]+'/LOGS1/'+history_filename
        data = GetColumnMESAhist(filename_history,col_bin)

        # Get some of the properties
        log_L_b = data[col_bin.index('log_L')]
        log_Lnuc = data[col_bin.index('log_Lnuc')]

        # locate the ZAMS
        ind_ZAMS_b = locate_ZAMS(log_L_b, log_Lnuc)   

        # Update the shape of the properties
        # Rewrite the properties to remove the pre-MS
        MESA_params_b[i] = [None]*nbr_MESA_param_b
        for cc in range(nbr_MESA_param_b):
            # Store the parameter from ZAMS and onwards
            MESA_params_b[i][cc] = data[cc][ind_ZAMS_b:]
            # Edit the overflow parameter 
            if col_bin[cc] == 'rl_relative_overflow_1':
                MESA_params_b[i][cc] = data[cc][ind_ZAMS_b:]>0.

        # Save properties
        indices = np.arange(len(MESA_params_b[i][0]))
        center_h1_b = MESA_params_b[i][col_bin.index('center_h1')]
        surface_h1_b = MESA_params_b[i][col_bin.index('surface_h1')]
        center_he4_b = MESA_params_b[i][col_bin.index('center_he4')]
        star_mass_b = MESA_params_b[i][col_bin.index('star_mass')]
        log_R_b = MESA_params_b[i][col_bin.index('log_R')]
        RLOF_b = MESA_params_b[i][col_bin.index('rl_relative_overflow_1')]
        star_age_b = MESA_params_b[i][col_bin.index('star_age')]
        
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

        # Tell the log that the model was read
        fid_log = open(log_name,'a')
        fid_log.write('At '+sims_bin[i]+'\n')
        fid_log.close()

    # Tell the log that the evolutionary models for stripped stars have been read
    fid_log = open(log_name,'a')
    fid_log.write('The evolutionary models for stripped stars have been read. \n')
    fid_log.close()



    if save_figs:
        # Test figures
        # # # # Plot the HRD # # # # # # # #
        fig, ax = plt.subplots(1,1,figsize=(8,6))
        for i in range(nbr_sims_bin):
            log_Teff_b = MESA_params_b[i][col_bin.index('log_Teff')]
            log_L_b = MESA_params_b[i][col_bin.index('log_L')]
            RLOF_b = MESA_params_b[i][col_bin.index('rl_relative_overflow_1')]
            if i == 0:
                ax.plot(log_Teff_b[RLOF_b[i]],log_L_b[RLOF_b[i]],'b-',lw=5, label='RLOF')
                ax.plot(log_Teff_b[ind_he[i]],log_L_b[ind_he[i]],'ro',label='ind He')
                ax.plot(log_Teff_b[ind_detach[i]],log_L_b[ind_detach[i]],'go',label='Detatchment')
                ax.plot(log_Teff_b[ind_end[i]],log_L_b[ind_end[i]],'mo',label='End He burning')
            else:
                ax.plot(log_Teff_b[RLOF_b[i]],log_L_b[RLOF_b[i]],'b-',lw=5)
                ax.plot(log_Teff_b[ind_he[i]],log_L_b[ind_he[i]],'ro')
                ax.plot(log_Teff_b[ind_detach[i]],log_L_b[ind_detach[i]],'go')
                ax.plot(log_Teff_b[ind_end[i]],log_L_b[ind_end[i]],'mo')
            ax.plot(log_Teff_b,log_L_b,'k-')
        ax.set_xlabel('$\\log_{10} (T_{\\mathrm{eff}}/\\mathrm{K})$')
        ax.set_ylabel('$\\log_{10} (L/L_{\\odot})$')
        ax.invert_xaxis()
        ax.legend(loc=0,fontsize=0.8*fsize)
        fig.savefig(plots_dir+'/HRD_stripped.png',format='png',bbox_inches='tight',pad_inches=0.1)
        plt.close(fig)
        # # # # # # # # # # # # # # # # # # 

        # # # # Minit with Mstrip # # # # # # # #
        fig, ax = plt.subplots(1,1,figsize=(8,6))
        ax.loglog(m_bin_grid,mstrip_grid,'.')
        ax.set_xlabel('Initial mass [$M_{\\odot}$]')
        ax.set_ylabel('Stripped star mass [$M_{\\odot}$]')
        fig.savefig(plots_dir+'/M_stripped.png',format='png',bbox_inches='tight',pad_inches=0.1)
        plt.close(fig)
        # # # # # # # # # # # # # # # # # # 

        # # # # Mstrip with Rstrip # # # # # # # #
        fig, ax = plt.subplots(1,1,figsize=(8,6))
        ax.loglog(mstrip_grid,rstrip_grid,'.')
        ax.set_xlabel('Stripped star mass [$M_{\\odot}$]')
        ax.set_ylabel('Radius of stripped star [$R_{\\odot}$]')
        fig.savefig(plots_dir+'/MR_stripped.png',format='png',bbox_inches='tight',pad_inches=0.1)
        plt.close(fig)
        # # # # # # # # # # # # # # # # # # 

        # # # # Mstrip with tau_strip # # # # # # # #
        ind_full_strip = strip_duration_grid > 0.
        fig, ax = plt.subplots(1,1,figsize=(8,6))
        ax.loglog(mstrip_grid[ind_full_strip],strip_duration_grid[ind_full_strip],'.')
        ax.set_xlabel('Stripped star mass [$M_{\\odot}$]')
        ax.set_ylabel('Duration of the stripped phase [yrs]')
        fig.savefig(plots_dir+'/Mtau_stripped.png',format='png',bbox_inches='tight',pad_inches=0.1)
        plt.close(fig)
        # # # # # # # # # # # # # # # # # # 


        # Tell the log that the evolutionary models for stripped stars have been read
        fid_log = open(log_name,'a')
        fid_log.write('Saved some figures in '+plots_dir+' \n\n')
        fid_log.close()
        
    

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                                                             #
#     P O P U L A T I O N   S Y N T H E S I S                                 #
#                                                                             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                                                             #
#     - Distributions                                                         #
#                                                                             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#
# Here is where the population synthesis is starting. 
#

# Tell the log that we are entering the population synthesis phase
fid_log = open(log_name,'a')
fid_log.write('Entering the population synthesis! \n')
fid_log.write('This is Monte Carlo\n\n')
fid_log.write('IMF: picking masses of stars\n')
fid_log.close()

# In case the star-formation rate is too high, we need to start a loop because python cannot handle that much data
num_turns = 1
if type_SF == 'constant':
    if starformation_rate > SFR_lim:
        num_turns = int(np.ceil(starformation_rate/SFR_lim))
    SFR_run = copy.copy(SFR_lim)

fid_log = open(log_name,'a')
fid_log.write('Going to run '+str(num_turns)+' loops\n')
fid_log.close()
        

# Start the loop (because SFR too high for memory)
for iii in range(num_turns):

    fid_log = open(log_name,'a')
    fid_log.write('Starting turn '+str(iii+1)+'..........\n')
    fid_log.close()
    
    # # # # # #  Initial Mass Function (IMF) # # # # # # 

    # Give a total mass of the population
    if type_SF == 'starburst':
        Target_M = copy.copy(total_mass_starburst)          # MSun
        nbr_stars = int(5*total_mass_starburst)             # this is just a guess, it is updated later (needs to be >actual nbr)
    elif type_SF == 'constant':
        rate = copy.copy(SFR_run)                                 # Star formation rate in MSun/yr
        Target_M = rate*duration
        evaluation_time = np.array([duration])
        nbr_stars = int(2.*rate*duration)

    # Random array - uniform distribution
    U = np.random.rand(nbr_stars)

    # Draw the masses of the stars
    if IMF_choice == 'Salpeter':
        m = Salpeter_IMF(U,mmin,mmax)
    elif IMF_choice == 'Kroupa':
        m = Kroupa_IMF(U,mmin,mmax)
    elif IMF_choice == 'Alt_Kroupa':
        m = Alt_Kroupa_IMF(U,mmin,mmax,alpha_IMF)
    mtmp = np.cumsum(m)
    ind = np.argmin(np.abs(mtmp - Target_M))
    Mtot = mtmp[ind]
    m = m[:ind]
    nbr_stars = len(m)
    

    # Tell the log that the masses have been picked
    fid_log = open(log_name,'a')
    fid_log.write('IMF gives a total stellar mass of '+str(Mtot)+' MSun and '+str(nbr_stars)+' number of stars \n')
    fid_log.close()


    if iii == 0:
        if save_figs:
            # # # # Plot the IMF # # # # # # # #
            fig, ax = plt.subplots(1,1,figsize=(6,4.5))
            ax.hist(np.log10(m),100,log=True)
            ax.set_xlabel('Mass [$M_{\\odot}$]')
            ax.set_ylabel('Number of stars')
            xtick = [0,1,2]
            ax.set_xticks(xtick)
            ax.set_xticklabels([1,10,100])
            for i in range(len(xtick)):
                ax.get_xaxis().majorTicks[i].set_pad(7)
            ax.tick_params('both', length=8, width=1.5, which='major')
            ax.tick_params('both', length=4, width=1.0, which='minor')
            ax.set_yticks([])
            ax.set_xlim(np.log10([mmin,mmax]))
            ax.tick_params(direction="in", which='both')
            fig.savefig(plots_dir+'/IMF.png',format='png',bbox_inches='tight',pad_inches=0.1)
            plt.close(fig)
            # # # # # # # # # # # # # # # # # # 

            # Tell the log
            fid_log = open(log_name,'a')
            fid_log.write('Saved a plot of the IMF distribution in '+plots_dir+' \n')
            fid_log.close()


    # Give the stars a birthday
    if type_SF == 'starburst':
        birthday = np.zeros(nbr_stars)
    elif type_SF == 'constant':
        birthday = np.random.random(nbr_stars)*duration

    # Tell the log
    fid_log = open(log_name,'a')
    fid_log.write('Assigned birthdays to the stars \n\n')
    fid_log.close()

    # # # # # #  Initial B-field Function (IBF) # # # # # #
    # 
    if compute_magnetic:

        # Make an array to hold the initial magnetic field strengths
        Binit = np.zeros(nbr_stars)

        # Pick randomly stars of a certain fraction to be magnetic
        ind_B = np.random.random(nbr_stars) <= Bfrac  
        nbr_B = np.sum(ind_B)

        # Use the function given in the input file
        if IBF_choice == 'IBF_flat':
            Binit[ind_B] = IBF_flat(nbr_B, Bmin, Bmax)
        elif IBF_choice == 'IBF_gaussian':
            Binit[ind_B] = IBF_gaussian(nbr_B, Bmin, Bmax, Bmean, Bstdev)  # I think this basically is a poisson distribution now
        elif IBF_choice == 'IBF_log10normal':
            Binit[ind_B] = IBF_log10normal(nbr_B, Bmin, Bmax, Bmean, Bstdev)  # Updated here to be normal-distributed in log10 instead of natural log.
        elif IBF_choice == 'IBF_poisson':
            Binit[ind_B] = IBF_poisson(nbr_B, k, mu)   # I don't think the k and mu work yet in the input file - check it.. 

        
        # This is the number of magnetic stars
        nbr_B = np.sum(Binit>0.)
            


        # Only for the first loop to not overload
        if iii == 0:
            if save_figs:
                # Plot the B-field distribution
                fig, ax = plt.subplots(1,1,figsize=(6,4.5))
                ax.hist(Binit,50)
                ax.set_xlabel('B-field [G]')
                ax.set_ylabel('Number of stars')
                ax.tick_params('both', length=8, width=1.5, which='major')
                ax.tick_params('both', length=4, width=1.0, which='minor')
                ax.set_yticks([])
                ax.set_xlim([Bmin,Bmax])
                if IBF_choice in ['IBF_gaussian','IBF_lognormal','IBF_poisson']:
                    ax.set_yscale('log')
                ax.tick_params(direction="in", which='both')
                fig.savefig(plots_dir+'/IBF.png',format='png',bbox_inches='tight',pad_inches=0.1)
                plt.close(fig)
                # # # # # # # # # # # # # # # # # #             

        # Tell the log
        fid_log = open(log_name,'a')
        fid_log.write('IBF: Assigned initial B-fields for the stars, using the '+IBF_choice+' function \n\n')
        fid_log.close()        

            
    
    if compute_binaries:

        # # # # # #  Binary fraction, f_bin # # # # # # 
        # The binary fraction is drawn from distributions

        # Tell the log
        fid_log = open(log_name,'a')
        fid_log.write('BINARY FRACTION: Going to pick which stars are in binaries \n')
        fid_log.close()

        # Duchene & Kraus (2013)
        if fbin_choice == 'DK':

            # Tell the log
            fid_log = open(log_name,'a')
            fid_log.write('Using the binary fraction from Duchene and Kraus (2013)\n')
            fid_log.close()

            # The binary fraction is mass dependent
            primary = np.zeros(nbr_stars)

            # Very low mass stars are not included in this analysis but Duchene & Kraus do mention
            # something about the companion fraction so just in case I put here that Aller (2007) 
            # estimated the multiplicity fraction for m <~ 0.1 MSun to be 22^{+6}_{-4} %. 

            # Low-mass stars data come from nearly complete samples as described in Duchene & Kraus (2013)
            # from Delfosse+ 04, Dieterich+ 12 and Reid & Gizis (97a)
            ind_low_mass = (m < 0.5)*(m >= 0.1)
            nbr_low_mass = np.sum(ind_low_mass)
            if nbr_low_mass>0:
                primary[ind_low_mass] = np.random.random(nbr_low_mass) <= 0.26

            # WHAT ABOUT 0.5 < m < 0.7 MSun ????? 

            # For the solar-mass stars I pick from Duchene & Kraus (2013) data from Raghavan+ 2010
            # 0.7 < m < 1.0 MSun
            ind_solar_mass1 = (m < 1.0)*(m >= 0.7)
            nbr_solar_mass1 = np.sum(ind_solar_mass1)
            if nbr_solar_mass1>0:
                primary[ind_solar_mass1] = np.random.random(nbr_solar_mass1) <= 0.41

            # 1.0 < m < 1.3 MSun
            ind_solar_mass2 = (m < 1.3)*(m >= 1.0)
            nbr_solar_mass2 = np.sum(ind_solar_mass2)
            if nbr_solar_mass2>0:
                primary[ind_solar_mass2] = np.random.random(nbr_solar_mass2) <= 0.5

            # WHAT ABOUT 1.3 < m < 1.5 MSun ??????

            # Intermediate mass stars 
            # Duchene & Kraus are not very precise here, but say the multiplicity fraction is > 0.5
            # 1.5 < m < 5.0 MSun
            ind_intermediate_mass = (m < 5.0)*(m >= 1.5)
            nbr_intermediate_mass = np.sum(ind_intermediate_mass)
            if nbr_intermediate_mass>0:
                primary[ind_intermediate_mass] = np.random.random(nbr_intermediate_mass) <= 0.5

            # High mass stars
            # For high-mass stars I will just bluntly use 0.7. It is very optimistic in Duchene & Kraus.
            ind_high_mass = m >= 5.0
            nbr_high_mass = np.sum(ind_high_mass)
            if nbr_high_mass>0:
                primary[ind_high_mass] = np.random.random(nbr_high_mass) <= 0.69

            # Total number of binary stars
            nbr_bin = np.sum(primary)

            # Make primary a boolean array
            primary = primary == 1


            if iii == 0:
                if save_figs:
                    # # # # Plot the fbin # # # # # # # #
                    mm = np.logspace(0,2.5,101)
                    fbin_plot = np.zeros(len(mm)-1)
                    mid_m = np.zeros(len(mm)-1)
                    for i in range(len(mm)-1):
                        bin_bin = (m[primary] >= mm[i])*(m[primary] < mm[i+1])
                        sin_bin = (m >= mm[i])*(m < mm[i+1])
                        if np.sum(sin_bin) >0:
                            fbin_plot[i] = float(np.sum(bin_bin))/float(np.sum(sin_bin))
                        mid_m[i] = mm[i]+(mm[i+1]-mm[i])/2.0

                    fig, ax = plt.subplots(1,1,figsize=(6,4.5))
                    ax.semilogx(mid_m,fbin_plot,'b-',lw=2)
                    ax.set_xlabel('Mass [$M_{\\odot}$]')
                    ax.set_ylabel('$f_{\\mathrm{bin}}$')
                    xtick = [1,10,100]
                    ax.set_xticks(xtick)
                    ax.set_xticklabels([1,10,100])
                    for i in range(len(xtick)):
                        ax.get_xaxis().majorTicks[i].set_pad(7)
                    ax.tick_params('both', length=8, width=1.5, which='major')
                    ax.tick_params('both', length=4, width=1.0, which='minor')
                    ytick = [0,0.2,0.4,0.6,0.8,1.0]
                    ax.set_yticks(ytick)
                    ax.set_xlim([mmin,mmax])
                    fig.savefig(plots_dir+'/fbin.png',format='png',bbox_inches='tight',pad_inches=0.1)
                    plt.close(fig)
                    # # # # # # # # # # # # # # # # # # 



        # The analytical relation of van Haaften et al. (2013)
        elif fbin_choice == 'vH13':

            # Tell the log
            fid_log = open(log_name,'a')
            fid_log.write('Using the binary fraction from van Haaften et al. (2013)\n')
            fid_log.close()

            # Create a random array 
            temp_random = np.random.random(nbr_stars)

            # Get the binary fraction for each mass
            fbin = 0.5 + 0.25*np.log10(m)
            fbin[fbin>1.0] = 1.0

            # Determine which stars are in binary
            primary = temp_random <= fbin

            # How many binaries is that?
            nbr_bin = np.sum(primary)

            if iii == 0:
                if save_figs:
                    # # # # Plot the fbin # # # # # # # #
                    clr_l = np.array([118, 68, 138])/255.
                    clr_f = np.array([175, 122, 197])/255.
                    mm = np.logspace(0,2.5,100)
                    fbin_plot = 0.5+0.25*np.log10(mm)
                    fbin_plot[fbin_plot>1.0] = 1.0
                    fig, ax = plt.subplots(1,1,figsize=(6,4.5))
                    ax.semilogx(mm,fbin_plot,'-',color=clr_l,lw=3)
                    ax.fill_between(mm,np.zeros(len(fbin_plot)),fbin_plot,color=clr_f)
                    ax.set_xlabel('Mass [$M_{\\odot}$]')
                    ax.set_ylabel('$f_{\\mathrm{bin}}$')
                    xtick = [1,10,100]
                    ax.set_xticks(xtick)
                    ax.set_xticklabels([1,10,100])
                    for i in range(len(xtick)):
                        ax.get_xaxis().majorTicks[i].set_pad(7)
                    ax.tick_params('both', length=8, width=1.5, which='major')
                    ax.tick_params('both', length=4, width=1.0, which='minor')
                    ax.set_xlim([1,100])
                    ax.set_ylim([0,1])
                    fig.savefig(plots_dir+'/fbin.png',format='png',bbox_inches='tight',pad_inches=0.1)
                    plt.close(fig)
                    # # # # # # # # # # # # # # # # # # 


        # This binary fraction is from Moe & DiStefano (2017), see their Fig. 42
        elif fbin_choice == 'moe':

            # Tell the log
            fid_log = open(log_name,'a')
            fid_log.write('Using a function fitted to Fig. 42 of Moe & DiStefano (2017) for the binary fraction \n')
            fid_log.close()

            # Get the binary fraction for each mass
            # steps: 
            # (1) read the Fig 42 of Moe & DiStefano (2017), 
            #filename_moe = '/data001/ygoetberg/taurus/python_scripts/ionisation_paper/moe_distefano_17.txt'
            #data = np.loadtxt(filename_moe)
            #M_tmp = data[:,0]
            #fbin_tmp = data[:,1]
            # This file contains the binary fractions for interacting binaries that Moe & DiStefano (2017) found (their Fig. 42), for 0.2 < log10 P < 3.7 and q > 0.1
            # I have used plot digitizer to get these values.
            M_tmp = np.array([28.208070697850278,12.194394238503687,7.013958043119244,3.5179000994706486,1.0033899071551038])
            fbin_tmp = np.array([1.039580885942532,0.7743880876109841,0.621660551424101,0.3699230117699013,0.1392481270452426])
            # (2) fit line, 
            coeff = np.polyfit(np.log10(M_tmp),fbin_tmp,1)
            # (3) use line as function
            fbin = np.polyval(coeff,np.log10(m))
            fbin[fbin>1.] = 1.

            # Create a random array
            temp_random = np.random.random(nbr_stars)

            # Determine which stars are in binary
            primary = temp_random <= fbin

            # How many binaries is that? 
            nbr_bin = np.sum(primary)

            if iii == 0:
                if save_figs:
                    # # # # Plot the fbin # # # # # # # #
                    clr_l = np.array([118, 68, 138])/255.
                    clr_f = np.array([175, 122, 197])/255.
                    mm = np.logspace(0,2.5,100)
                    fbin_plot = coeff[0]*np.log10(mm) + coeff[1]
                    fbin_plot[fbin_plot>1.0] = 1.0
                    fig, ax = plt.subplots(1,1,figsize=(6,4.5))
                    ax.semilogx(mm,fbin_plot,'-',color=clr_l,lw=3)
                    ax.fill_between(mm,np.zeros(len(fbin_plot)),fbin_plot,color=clr_f)
                    ax.set_xlabel('Mass [$M_{\\odot}$]')
                    ax.set_ylabel('$f_{\\mathrm{bin}}$')
                    xtick = [1,10,100]
                    ax.set_xticks(xtick)
                    ax.set_xticklabels([1,10,100])
                    for i in range(len(xtick)):
                        ax.get_xaxis().majorTicks[i].set_pad(7)
                    ax.tick_params('both', length=8, width=1.5, which='major')
                    ax.tick_params('both', length=4, width=1.0, which='minor')
                    ax.set_xlim([1,100])
                    ax.set_ylim([0,1.05])
                    ax.tick_params(direction="in", which='both')
                    fig.savefig(plots_dir+'/fbin.png',format='png',bbox_inches='tight',pad_inches=0.1)
                    plt.close(fig)
                    # # # # # # # # # # # # # # # # # # 


        # Allow for a mass-independent binary fraction too
        elif fbin_choice == 'constant':

            # Tell the log
            fid_log = open(log_name,'a')
            fid_log.write('Assuming a mass-independent binary fraction of fbin = '+str(fbin_constant)+'\n')
            fid_log.close()    

            # Create a random array
            temp_random = np.random.random(nbr_stars)

            # Determine which stars are in binary
            primary = temp_random <= fbin_constant

            # How many binaries is that? 
            nbr_bin = np.sum(primary)

            if iii == 0:
                if save_figs:
                    # # # # Plot the fbin # # # # # # # #
                    clr_l = np.array([118, 68, 138])/255.
                    clr_f = np.array([175, 122, 197])/255.
                    fig, ax = plt.subplots(1,1,figsize=(6,4.5))
                    medges = np.logspace(np.log10(mmin), np.log10(mmax),21)
                    mmid = 10**(np.log10(medges[:-1])+ (np.log10(medges[1:])-np.log10(medges[:-1]))/2.)
                    fbin_plot = np.zeros(len(medges)-1)
                    for k in range(len(medges)-1):
                        ind_mbin = (m>=medges[k])*(m<medges[k+1])
                        fbin_plot[k] = np.sum(primary*ind_mbin)/np.float_(np.sum(ind_mbin))
                    ax.semilogx(mmid,fbin_plot,'-',color=clr_l)
                    ax.semilogx(mmid,fbin_plot,'.',color=clr_f)
                    ax.set_xlabel('Mass [$M_{\\odot}$]')
                    ax.set_ylabel('$f_{\\mathrm{bin}}$')
                    xtick = [1,10,100]
                    ax.set_xticks(xtick)
                    ax.set_xticklabels([1,10,100])
                    for i in range(len(xtick)):
                        ax.get_xaxis().majorTicks[i].set_pad(7)
                    ax.tick_params('both', length=8, width=1.5, which='major')
                    ax.tick_params('both', length=4, width=1.0, which='minor')
                    ax.set_xlim([1,100])
                    ax.set_ylim([0,1.05])
                    ax.tick_params(direction="in", which='both')
                    fig.savefig(plots_dir+'/fbin.png',format='png',bbox_inches='tight',pad_inches=0.1)
                    plt.close(fig)
                    # # # # # # # # # # # # # # # # # # 

    # Not necessary to do the above if there are only single stars
    else: 
        nbr_bin = 0.
        primary = np.zeros(nbr_stars) != 0.

    single = primary==False

    # Tell the log
    fid_log = open(log_name,'a')
    fid_log.write('Number of binaries: '+ str(nbr_bin)+ ' (not corrected for total mass yet) \n')
    if save_figs:
        fid_log.write('Saved a figure describing the binary fraction in '+plots_dir+' and if binaries are included\n')
    fid_log.close()




    if compute_binaries:
        
        
        # # # # # #  Mass ratio, q # # # # # # 
        #
        # The mass ratio is drawn from distributions

        # Tell the log
        fid_log = open(log_name,'a')
        fid_log.write('\nMASS RATIO: Choosing masses of companions by randomly drawing from a mass ratio distribution \n')
        fid_log.close()

        # The flat mass ratio distribution
        if q_choice == 'flat':

            # Mass ratio is a tricky business and so far flat between 0.1 and 1 is pretty standard.
            q = (qmax-qmin)*np.random.random(nbr_bin) + qmin

            # Tell the log
            fid_log = open(log_name,'a')
            fid_log.write('Assigned masses to the secondaries following a flat distribution in mass ratio. \n')
            fid_log.close()


        # This is the power-law slope:    dN/dq = k*q^eta_q
        elif q_choice == 'power_slope':

            # Random array between 0 and 1
            U = np.random.random(nbr_bin)

            # Solving the formel
            q = (qmin**(eta_q+1.) + U*(qmax**(eta_q+1.) - qmin**(eta_q+1.)))**(1./(eta_q+1.))

            # Tell the log
            fid_log = open(log_name,'a')
            fid_log.write('Assigned masses to the secondaries following a power-law distribution in mass ratio with exponent'+str(eta_q)+'. \n')
            fid_log.close()


        # This is for 1D polynomial:    dN/dq = eta_q*q + bb 
        elif q_choice == 'linear_slope':

            # Ok, now I will use a 1D polynomial to draw mass ratios from
            # The distribution goes as dN/dq = eta_q*q + bb
            # Find the bb that normalizes the distribution
            bb = (1. - (eta_q/2.)*((qmax**2.) - (qmin**2.)))/(qmax-qmin)

            # This is the random numbers between 0 and 1, they will be used for when inverting the CDF to get the q
            U = np.random.random(nbr_bin)

            # Now solving the pq formel
            DD = -(eta_q/2.)*(qmin**2.) - bb*qmin - U
            EE = (2./eta_q)*DD
            q_minus = -(bb/eta_q) - np.sqrt((bb/eta_q)**2. - EE)
            q_plus = -(bb/eta_q) + np.sqrt((bb/eta_q)**2. - EE)
            q = np.zeros(nbr_bin)
            ind_qminus = (q_minus < qmax)*(q_minus > qmin)*(((q_plus < qmin)+(q_plus > qmax))>0.)
            ind_qplus = (q_plus < qmax)*(q_plus > qmin)*(((q_minus < qmin)+(q_minus > qmax))>0.)
            q[ind_qminus] = q_minus[ind_qminus]
            q[ind_qplus] = q_plus[ind_qplus]

            # Tell the log
            fid_log = open(log_name,'a')
            fid_log.write('Assigned masses to the secondaries following a 1D polynomial distribution in mass ratio: dN/dq = '+str(eta_q)+'q + bb. \n')
            fid_log.close()


        # Assign masses to the secondaries
        m2 = q*m[primary]

        # Now the total mass is too high -> scale down again to the total mass we should have.
        Mtot1 = np.sum(m)
        Mtot2 = np.sum(m2)
        Mtot = Mtot1 + Mtot2
        n = 0
        n2 = 0
        while Mtot > Target_M:
            Mtot1 = Mtot1-m[n]
            if primary[n]: 
                Mtot2 = Mtot2 - m2[n2]
                n2 = n2+1
            n = n+1
            Mtot = Mtot1+Mtot2
        m = m[n:]
        primary = primary[n:]
        single = single[n:]
        birthday = birthday[n:]
        # Total number of binary stars
        nbr_bin = np.sum(primary)
        # Total number of stars (counting binaries as 1 star)
        nbr_stars = len(m)

        # Update initial masses of primaries (m1) and secondaries (m2), initial mass ratio (q)
        m1 = m[primary]
        m2 = m2[n2:]
        q = q[n2:]
        # Set the birthday of the secondary to the same as the primary
        birthday_m2 = copy.copy(birthday[primary])

        # Tell the log
        fid_log = open(log_name,'a')
        fid_log.write('Removed some random stars so that the total mass is what we want. \n')
        fid_log.write('Total mass: '+str(Mtot)+' MSun, '+str(nbr_bin)+' binaries \n')
        fid_log.write('Set the birthday of the secondaries to the same as the primaries \n')
        fid_log.close()


        if iii == 0:
            if save_figs:
                # # # # Plot the q # # # # # # # #
                fig, ax = plt.subplots(1,1,figsize=(6,4.5))
                ax.hist(q,100)
                ax.set_xlabel('Mass ratio, $q$')
                ax.set_ylabel('Number stars')
                ax.set_yticks([])
                xtick = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                ax.set_xticks(xtick)
                ax.set_xticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                for i in range(len(xtick)):
                    ax.get_xaxis().majorTicks[i].set_pad(7)
                ax.tick_params('both', length=8, width=1.5, which='major')
                ax.tick_params('both', length=4, width=1.0, which='minor')
                ax.tick_params(direction="in", which='both')
                fig.savefig(plots_dir+'/q.png',format='png',bbox_inches='tight',pad_inches=0.1)
                plt.close(fig)
                # # # # # # # # # # # # # # # # # # 


                # # # Check the IMF again! # # #
                fig, ax1 = plt.subplots(1,1,figsize=(8,6))
                ax1.hist(np.log10(np.concatenate([m,m2])),100,log=True,label='IMF including M2')
                ax1.hist(np.log10(m),100,log=True,color='r',alpha=0.5,label='IMF excluding M2')
                ax1.set_xlabel('$\\log_{10} (M/M_{\\odot})$')
                ax1.set_ylabel('Number of stars')
                ax1.tick_params(direction="in", which='both')
                ax1.legend(loc=0,fontsize=0.8*fsize)
                fig.savefig(plots_dir+'/IMF_update_after_q.png',format='png',bbox_inches='tight',pad_inches=0.1)
                plt.close(fig)
                # # # # # # # # # # # # # # # # #



        # # # # # #  Period, P # # # # # # 
        # 
        # Period is picked randomly from distributions 

        # Tell the log
        fid_log = open(log_name,'a')
        fid_log.write('\nPERIOD: Going to assign orbital periods to the systems... \n')
        fid_log.close()

        # Period limits:
        # Pmin is P_ZAMS 
        # Pmax = 10^3.5 or 10^3.7 days
        # Starting from radius limits

        # Interpolate using the numpy
        R_ZAMS = np.interp(m1,m_grid,R_ZAMS_grid)
        R_ZAMS = extrapolate(m1,R_ZAMS,m_grid,R_ZAMS_grid)

        # Translate the above radii to periods at which the 
        # star would fill its Roche Lobe
        q_inv = 1.0/q    # This m1/m2
        rL = 0.49*(q_inv**(2.0/3.0))/(0.69*(q_inv**(2.0/3.0)) + np.log(1.0+(q_inv**(1.0/3.0))))

        # Period needed to fill Roche Lobe at ZAMS
        a_tmp = (R_ZAMS/rL)*RSun_AU  # separation in AU
        P_ZAMS = (4.0*(np.pi**2.0)*(a_tmp**3.0)/(G*(m1+m2)))**0.5    # Period in days

        # Now to the period limits
        P_min = copy.copy(P_ZAMS)        # in Sana+12 this is 10^0.15
        # The maximum period is set at the beginning. 


        # The combination of Opik (1924) and Sana et al. (2012) 
        if P_choice == 'Opik_Sana':

            # Period distribution. This is mass dependent. 
            P = np.zeros(nbr_bin)

            # # # # Massive stars
            # Sana distribution (Sana+ 12):   dN/d(log P) = k*(log P)^-0.55

            # When to apply the Sana period distribution? At O-star masses
            ind_Sana = m1 >= Mlim_Sana
            nbr_Sana = np.sum(ind_Sana)

            # Draw from the distribution
            uu = np.random.random(nbr_Sana)
            kappa_P_Sana = -0.55    # this is from Sana+12
            smin = np.log10(P_min[ind_Sana])
            smin[smin < 0.15] = 0.15   # this is also from Sana+12 and lower doesn't work because of maths
            smax = np.log10(P_max)
            s = (smin**(kappa_P_Sana+1.) + uu*(smax**(kappa_P_Sana+1.) - smin**(kappa_P_Sana+1.)))**(1./(kappa_P_Sana+1.))
            P_Sana = 10**s
            P[ind_Sana] = P_Sana


            # # # # Lower masses
            # Opik's law -- I choose all other masses in this period distribution
            ind_Opik = ind_Sana == 0
            nbr_Opik = np.sum(ind_Opik)

            # Opik's law is flat in log P space
            uu = np.random.random(nbr_Opik)
            P_Opik = 10**((np.log10(P_max)-np.log10(P_min[ind_Opik]))*uu + np.log10(P_min[ind_Opik]))
            P[ind_Opik] = P_Opik

            if iii == 0:
                if save_figs:
                    # # # # Plot the Period # # # # # # # #
                    clr_l = np.array([ 212, 172, 13 ])/255.
                    clr_h = np.array([211, 84, 0])/255.
                    fig, ax = plt.subplots(1,1,figsize=(6,4.5))
                    ax.hist(np.log10(P[m1>Mlim_Sana]),100,color=clr_h,edgecolor='none')
                    ax.hist(np.log10(P[m1<Mlim_Sana][0:np.sum(m1>15)]),100,color=clr_l,edgecolor='none',alpha=0.7)
                    ax.set_xlabel('Period [days]')
                    ax.set_ylabel('Number of stars')
                    xtick = [0,1,2,3]
                    ax.set_xticks(xtick)
                    ax.set_xticklabels([1, 10, 100, 1000])
                    for i in range(len(xtick)):
                        ax.get_xaxis().majorTicks[i].set_pad(7)
                    ax.tick_params('both', length=8, width=1.5, which='major')
                    ax.tick_params('both', length=4, width=1.0, which='minor')
                    fig.savefig(plots_dir+'/P_fewstars.png',format='png',bbox_inches='tight',pad_inches=0.1)
                    plt.close(fig)
                    # # # # # # # # # # # # # # # # #


            # Tell the log
            fid_log = open(log_name,'a')
            fid_log.write('Used the combination of Opik and Sana+12 for the period distribution\n')
            fid_log.write('The minimum period for the Sana+12 distribution is '+str(10**0.15)+'days, while the Opik distribution goes to minimum possible period.\n')
            fid_log.write('Ignoring widening via wind mass loss.\n')
            if save_figs:
                fid_log.write('Saved a plot in '+plots_dir+'\n\n')
            fid_log.close()


        # Power law:  dN/d(log P) = k*(log P)^kappa_P
        elif P_choice == 'log_power':

            # Period distribution.
            uu = np.random.random(nbr_bin)
            smin = np.log10(P_min)
            smin[smin < 0.15] = 0.15   # This is so that the function below works (~1.4 days)
            smax = np.log10(P_max)
            s = (smin**(kappa_P+1.) + uu*(smax**(kappa_P+1.) - smin**(kappa_P+1.)))**(1./(kappa_P+1.))
            P = 10**s       


            if iii == 0:
                if save_figs:
                    # # # # Plot the Period # # # # # # # #
                    fig, ax = plt.subplots(1,1,figsize=(6,4.5))
                    ax.hist(np.log10(P),100,color='b',edgecolor='none')
                    ax.set_xlabel('Period [days]')
                    ax.set_ylabel('Number of stars')
                    xtick = [0,1,2,3]
                    ax.set_xticks(xtick)
                    ax.set_xticklabels([1, 10, 100, 1000])
                    for i in range(len(xtick)):
                        ax.get_xaxis().majorTicks[i].set_pad(7)
                    ax.tick_params('both', length=8, width=1.5, which='major')
                    ax.tick_params('both', length=4, width=1.0, which='minor')
                    fig.savefig(plots_dir+'/P.png',format='png',bbox_inches='tight',pad_inches=0.1)
                    plt.close(fig)
                    # # # # # # # # # # # # # # # # #


            # Tell the log
            fid_log = open(log_name,'a')
            fid_log.write('Used the a power-law distribution that scales as ~ (log P)^'+str(kappa_P)+'\n')
            fid_log.write('Ignoring widening via wind mass loss.\n')
            if save_figs:
                fid_log.write('Saved a plot in '+plots_dir+'\n\n')
            fid_log.close()


        # Flat in log P (meaning favors short periods still, Opik 24)
        elif P_choice == 'log_flat':

            # Period distribution. This is mass dependent. 
            P = 10**(np.log10(P_min)+(np.log10(P_max)-np.log10(P_min))*np.random.random(nbr_bin))

            if iii == 0:
                if save_figs:
                    # # # # Plot the Period # # # # # # # #
                    fig, ax = plt.subplots(1,1,figsize=(6,4.5))
                    ax.hist(np.log10(P),100,color='b',edgecolor='none')
                    ax.set_xlabel('Period [days]')
                    ax.set_ylabel('Number of stars')
                    xtick = [0,1,2,3]
                    ax.set_xticks(xtick)
                    ax.set_xticklabels([1, 10, 100, 1000])
                    for i in range(len(xtick)):
                        ax.get_xaxis().majorTicks[i].set_pad(7)
                    ax.tick_params('both', length=8, width=1.5, which='major')
                    ax.tick_params('both', length=4, width=1.0, which='minor')
                    fig.savefig(plots_dir+'/P.png',format='png',bbox_inches='tight',pad_inches=0.1)
                    plt.close(fig)
                    # # # # # # # # # # # # # # # # #


            # Tell the log
            fid_log = open(log_name,'a')
            fid_log.write('Used the a power-law distribution that scales as ~ (log P)^0\n')
            fid_log.write('Ignoring widening via wind mass loss.\n')
            if save_figs:
                fid_log.write('Saved a plot in '+plots_dir+'\n\n')
            fid_log.close()
    
    



        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        #                                                                             #
        #     - Result of interaction                                                 #
        #                                                                             #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     
        #
        # Check which systems that will interact how (Case A, B, C and CEE)


        # Assign masses of the stripped stars
        # !!!! HERE: should we use core masses instead of extrapolating? !!!! 
        # !!!! HERE: what should we do with the case A systems? !!!!
        mstrip = 10**np.interp(np.log10(m1), np.log10(m_bin_grid), np.log10(mstrip_grid))
        mstrip = 10**extrapolate(np.log10(m1), np.log10(mstrip), np.log10(m_bin_grid), np.log10(mstrip_grid))
        # Instead of extrapolating these masses, let's use the helium core mass
        ind_massive_stripped = m1>np.max(m_bin_grid)
        mstrip[ind_massive_stripped] = 10**np.interp(np.log10(m1[ind_massive_stripped]), np.log10(m_grid), np.log10(he_core_mass_TAMS)) 

        # Tell the log
        fid_log = open(log_name,'a')
        fid_log.write('Interpolation: Assigned masses to the stripped stars using interpolation in log10 for both mstrip and minit.\n')
        fid_log.write('Instead of extrapolating, we use the He-core masses at TAMS for massive stripped stars\n')
        fid_log.write('The code cannot handle Case A stripped star masses yet - simply assumes it is as stable Case B for all...\n\n')
        fid_log.close()


        # Going to update the period distribution
        Pfinal = copy.copy(P)

        # Interpolate using the numpy (radii in RSun)
        R_ZAMS = np.interp(m1, m_grid, R_ZAMS_grid)
        R_ZAMS = extrapolate(m1, R_ZAMS, m_grid, R_ZAMS_grid)
        R_maxMS = np.interp(m1, m_grid, R_maxMS_grid)
        R_maxMS = extrapolate(m1, R_maxMS, m_grid, R_maxMS_grid)
        ind_tmp = R_maxMS < 0.
        R_maxMS[ind_tmp] = R_ZAMS[ind_tmp]
        R_conv = np.interp(m1, m_grid, R_conv_grid)
        R_conv = extrapolate(m1, R_conv, m_grid, R_conv_grid)
        ind_tmp = R_conv < 0.
        R_conv[ind_tmp] = R_maxMS[ind_tmp]
        R_maxHG = np.interp(m1, m_grid, R_maxHG_grid)
        R_maxHG = extrapolate(m1, R_maxHG, m_grid, R_maxHG_grid)
        ind_tmp = R_maxHG < 0.
        R_maxHG[ind_tmp] = R_conv[ind_tmp]

        # Tell the log
        fid_log = open(log_name,'a')
        fid_log.write('Interpolation: The radius limits from the single star grid linearly with the initial masses\n')
        fid_log.close()

        # Translate the above radii to periods at which the 
        # star would fill its Roche Lobe
        q_inv = 1.0/q    # This m1/m2
        rL = 0.49*(q_inv**(2.0/3.0))/(0.69*(q_inv**(2.0/3.0)) + np.log(1.0+(q_inv**(1.0/3.0))))

        # Period needed to fill Roche Lobe at ZAMS
        a_tmp = (R_ZAMS/rL)*RSun_AU  # separation in AU
        P_ZAMS = (4.0*(np.pi**2.0)*(a_tmp**3.0)/(G*(m1+m2)))**0.5    # Period in days

        # Period needed to fill Roche Lobe at TAMS
        a_tmp = (R_maxMS/rL)*RSun_AU
        P_maxMS = (4.0*(np.pi**2.0)*(a_tmp**3.0)/(G*(m1+m2)))**0.5    # Period in days

        # Period needed to fill Roche Lobe once a deep convective envelope has developed
        a_tmp = (R_conv/rL)*RSun_AU
        P_conv = (4.0*(np.pi**2.0)*(a_tmp**3.0)/(G*(m1+m2)))**0.5    # Period in days

        # Longest period needed to stil fill Roche Lobe during Hertzsprung gap
        a_tmp = (R_maxHG/rL)*RSun_AU
        P_maxHG = (4.0*(np.pi**2.0)*(a_tmp**3.0)/(G*(m1+m2)))**0.5    # Period in days

        # Tell the log
        fid_log = open(log_name,'a')
        fid_log.write('Translated the radius limits to period limits using the Roche potential\n')
        fid_log.close()


        # # # # # Evolutionary channels and stripping set here

        # Evolutionary channels
        # Mergers
        ind_mergeA = (((P < P_ZAMS)+(P < P_min_crit) > 0)+                              # too short period
                     ((P >= P_ZAMS)*(P >= P_min_crit)*(P <= P_maxMS)*(q<q_crit_MS)))     # too small q, case A
        ind_mergeC = ((P > P_maxHG)*(q<q_crit_HG))                                      # too small q, case C


        # Case A stable mass transfer
        ind_caseA = (P >= P_ZAMS)*(P >= P_min_crit)*(P <= P_maxMS)*(q>=q_crit_MS)       # Stable RLOF, case A

        # Case B is complicated because of the convective envelope
        Ptmp = copy.copy(P_maxHG)
        indtmp = P_conv < P_maxHG
        Ptmp[indtmp] = P_conv[indtmp]

        ind_caseB = (P > P_maxMS)*(P <= Ptmp)*(q>=q_crit_HG)                            # Stable RLOF, case B

        # Tell the log
        fid_log = open(log_name,'a')
        fid_log.write('Systems that go through stable Case A and stable Case B assigned... \n')
        fid_log.write('Systems that merge during main sequence and after helium burnign assigned... \n')
        fid_log.close()



        # I am including a possibility to use the alpha-prescription (given at beginning)
        if alpha_prescription: 

            # Tell the log
            fid_log = open(log_name,'a')
            fid_log.write('Entering the alpha-prescription \n')
            fid_log.close()

            # alpha-prescription is alpha = Delta_Ebind / Delta_Eorb

            # For which stars is this relevant?
            # - stars that enter a common envelope phase during the Hertzsprung gap
            #     o donor star has a convective envelope
            #     o the mass ratio is smaller than q_crit
            ind_donor_convective = (P > Ptmp)*(P <= P_maxHG)
            ind_small_q_HG = (P > P_maxMS)*(P <= Ptmp)*(q < q_crit_HG)
            ind_alpha_pres = (ind_donor_convective + ind_small_q_HG) > 0

            # Tell the log
            fid_log = open(log_name,'a')
            fid_log.write('Assuming that common envelope is initiated either because q < q_crit_HG or the donor has a convective envelope \n')
            fid_log.close()


            # Calculate Delta_Ebind
            # Need the mass of the envelopes
            menv = m1 - mstrip   # MSun
            # And the Roche radii of the primaries (prior to interaction)
            ainit = ((P**2.)*((G*(m1+m2))/(4.*np.pi**2.)))**(1./3.)  # AU
            RL1 = rL*ainit   # AU
            # Calculate the difference in binding energy (1. envelope is there, 2. envelope is not there)
            Delta_Ebind = -G*m1*menv/(lambda_CE*RL1)   # MSun (AU/day)^2

            # With the assumption for the alpha_CE and lambda_CE given earlier, I turn around the
            # equation to calculate the resulting separation between the two stars
            afinal = mstrip*m2/((m1*m2/ainit) - (2.*Delta_Ebind/(G*alpha_CE)))   # AU


            # The radius of the stripped stars are interpolated using the models
            rstrip = 10**np.interp(np.log10(mstrip),np.log10(mstrip_grid),np.log10(rstrip_grid))
            rstrip = 10**extrapolate(np.log10(mstrip),np.log10(rstrip),np.log10(mstrip_grid),np.log10(rstrip_grid))
            #rstrip = rstrip*RSun_AU   # now in AU
            rstrip = rstrip*u.R_sun.to(u.AU)   # now in AU

            # Tell the log
            fid_log = open(log_name,'a')
            fid_log.write('Interpolation: The radii of the stripped stars were interpolated and extrapolated using log10 of mstrip and log10 of rstrip \n')
            fid_log.close()    


            # Check if the stars fit within the orbit: 
            #  a is semi-major axis of the circle where one star is static the other moving around the first 
            #  - i.e., the separation
            # The radius of the companion star is less than the TAMS radius of its initial mass 
            #  (assuming no mass accretion since it goes through common envelope)
            rm2_lim = np.interp(m2,m_grid,R_ZAMS_grid)   # RSun
            rm2_lim = (extrapolate(m2, rm2_lim, m_grid, R_ZAMS_grid))*RSun_AU  # AU
            # Calculate the Roche-radii for both stars
            q1 = mstrip/m2
            rL_strip = 0.49*(q1**(2./3.))/(0.69*(q1**(2./3.)) + np.log(1.+(q1**(1./3.))))
            RL_strip = rL_strip*afinal   # AU
            q2 = m2/mstrip
            rL_2 = 0.49*(q2**(2./3.))/(0.69*(q2**(2./3.)) + np.log(1.+(q2**(1./3.))))
            RL_2 = rL_2*afinal   # AU

            # For the systems that fit in the orbit, none of the stars fill the Roche lobe
            ind_fit_in_orbit = ((rm2_lim < RL_2)*(rstrip < RL_strip)) > 0.

            # The stars that successfully eject the common envelope:
            ind_caseB_CEE = (ind_fit_in_orbit * ind_alpha_pres) > 0.

            # Tell the log
            fid_log = open(log_name,'a')
            fid_log.write('Assuming that a binary survives common envelope if the two (assuming circular) stars both fit in the orbit.\n')
            fid_log.write('It has then created a stripped star\n')
            fid_log.close() 

            # The stars that merge during the common envelope initiated during HG evolution of the donor
            ind_mergeB = ((ind_fit_in_orbit == False)*ind_alpha_pres) > 0.

            # Calculate the final period in days
            Pfinal[ind_caseB_CEE] = ((4.0*(np.pi**2.0)*
                                      (afinal[ind_caseB_CEE]**3.0)/(G*(mstrip[ind_caseB_CEE]+m2[ind_caseB_CEE])))**0.5)

            # Tell the log
            fid_log = open(log_name,'a')
            fid_log.write('Calculated the periods of the stripped star + companion systems that were created through CEE\n')
            fid_log.close() 


        # If not using the alpha-prescription, follow roughly the figure of Manos (binary_c predictions + analytical)
        else:
            ind_caseB_CEE = (P > Ptmp)*(P <= P_maxHG)*(q>=q_crit_HG)                       # Unstable RLOF -> CEE, case B
            ind_mergeB = ((P > P_maxMS)*(P <= P_maxHG)*(q<q_crit_HG))                       # too small q, case B

            # Tell the log
            fid_log = open(log_name,'a')
            fid_log.write('NOT USING ALPHA-PRESCRIPTION: simple assumptions for which stars merge or successfully eject the envelope.\n')
            fid_log.close() 


        # Case C (usually unstable, but also super short-lived)
        ind_caseC = (P > P_maxHG)*(q>=q_crit_HG)                                       # case C -> uncertain outcome

        # Tell the log
        fid_log = open(log_name,'a')
        fid_log.write('Assigned which stars enter mass transfer after helium burning (Case C) - no assumptions for what happen to these - too short-lived.\n')
        fid_log.close() 

        # This index is for the mass range that we consider for interaction, in general
        ind_interaction_mass = (m1 <= Minit_strip_max)*(m1 >= Minit_strip_min)  # the stars that can strip



        # # # # # LIMITS to which stars we consider
        # 
        # Separate out the mass range we want to consider for the different interactions
        # - we only want to consider stars that have an initial mass between the limits in the ind_interaction_mass

        # I am going to cut out the interacting systems that we consider
        # these indices have the size of all binaries  

        # Stripped star systems
        # long-lasting
        ind_caseA = ind_caseA*ind_interaction_mass     # RLOF on MS evolution of the donor
        ind_caseB = ind_caseB*ind_interaction_mass     # RLOF on HG evolution of the donor
        ind_caseB_CEE = ind_caseB_CEE*ind_interaction_mass   # CEE leading to envelope-stripping, donor is on HG
        # short-lasting
        ind_caseC = ind_caseC*ind_interaction_mass     # RLOF after HG evolution of the donor, using q>q_crit_HG

        # Systems that go through a merger
        ind_mergeA = ind_mergeA*ind_interaction_mass   # Merger of two MS stars
        ind_mergeB = ind_mergeB*ind_interaction_mass   # Merger of XX+HG system, did not check what the evol state of M2 is.
        ind_mergeC = ind_mergeC*ind_interaction_mass   # Merger of XX+post-HG system



        # General indices for the binary products
        ind_strip = (ind_caseA+ind_caseB+ind_caseB_CEE)>0
        ind_merge = (ind_mergeA+ind_mergeB)>0

        # Tell the log
        fid_log = open(log_name,'a')
        fid_log.write('Updated which stars that interact in which way... \n')
        fid_log.close() 



    # # # # # LIFETIMES of the stars
    # (back to both singles and binaries)

    if compute_magnetic == False:
        ind_life = lifetime_grid > 0.
        if compute_binaries:
            ind_full_strip = strip_duration_grid > 0.

        # Get the total lifetimes of single stars - calculated from single stellar models
        t_lifetime_m = 10**np.interp(np.log10(m), np.log10(m_grid[ind_life]), np.log10(lifetime_grid[ind_life]))
        t_lifetime_m = 10**extrapolate(np.log10(m), np.log10(t_lifetime_m), 
                                       np.log10(m_grid[ind_life]), np.log10(lifetime_grid[ind_life]))

    else:
        
        # 1st 2D interpolation - easy one - the lifetime of the stars
        # Get the grid to interpolate over - no changes, assuming it is smooth
        # Masses
        m_grid_B = np.array(B_m_grid[0])
        
        # Create the interpolation function
        z = np.log10(np.array(lifetime_grid_B))
        f = interpolate.RegularGridInterpolator((np.log10(m_grid_B),B_strength_grids), z.T)

        # Make the operation
        # It only works for stars within the interpolation ranges
        ind_m = (m>=np.min(m_grid_B))*(m<=np.max(m_grid_B))*(Binit>=np.min(B_strength_grids))*(Binit<=np.max(B_strength_grids))
        xy_new = list(zip(np.log10(m[ind_m]),Binit[ind_m]))
        t_lifetime_m = np.zeros(nbr_stars)
        t_lifetime_m[ind_m] = 10**np.array(f(xy_new))
        # Disrespect the magnetic field for the extrapolation
        t_lifetime_m[m<np.min(m_grid_B)] = np.max(t_lifetime_m[ind_m])
        t_lifetime_m[m>np.max(m_grid_B)] = np.min(t_lifetime_m[ind_m])
        # No extrapolation... 
        #t_lifetime_m[ind_mo] = 10**extrapolate(np.log10(m[ind_mo]), np.log10(t_lifetime_m[ind_mo]), 
        #                               np.log10(m_grid_B), np.log10(lifetime_grid_B[0]))

        
        # Save a figure with this
        if iii == 0:
            if save_figs:
                # # # # Plot the lifetime # # # # # # # #
                fig, ax = plt.subplots(1,1,figsize=(6,4.5))
                sc = ax.scatter(np.log10(m),Binit,c=np.log10(t_lifetime_m))
                plt.colorbar(sc)
                ax.set_xlabel('log10 mass [Msun]')
                ax.set_ylabel('Binit [G]')
                ax.tick_params('both', length=8, width=1.5, which='major')
                ax.tick_params('both', length=4, width=1.0, which='minor')
                fig.savefig(plots_dir+'/tau_B.png',format='png',bbox_inches='tight',pad_inches=0.1)
                plt.close(fig)
                # # # # # # # # # # # # # # # # #
            
            # Tell the log about the success. 
            fid_log = open(log_name,'a')
            fid_log.write('First 2D interpolation done! - lifetimes \n')
            fid_log.close()
        
        
    if compute_binaries:
        t_lifetime_m2 = 10**np.interp(np.log10(m2), np.log10(m_grid[ind_life]), np.log10(lifetime_grid[ind_life]))
        t_lifetime_m2 = 10**extrapolate(np.log10(m2), np.log10(t_lifetime_m2), 
                                        np.log10(m_grid[ind_life]), np.log10(lifetime_grid[ind_life]))

    # Tell the log
    fid_log = open(log_name,'a')
    fid_log.write('Interpolation: Assigned lifetimes of the stars (single stellar grid) \n')
    fid_log.close() 

    if compute_magnetic == False:
        # Main sequence lifetimes [yrs]
        t_MS_m = 10**np.interp(np.log10(m), np.log10(m_grid), np.log10(MS_duration_grid))
        t_MS_m = 10**extrapolate(np.log10(m), np.log10(t_MS_m), np.log10(m_grid), np.log10(MS_duration_grid))
        if compute_binaries:
            t_MS_m2 = 10**np.interp(np.log10(m2), np.log10(m_grid), np.log10(MS_duration_grid))
            t_MS_m2 = 10**extrapolate(np.log10(m2), np.log10(t_MS_m2), np.log10(m_grid), np.log10(MS_duration_grid))

        # Tell the log
        fid_log = open(log_name,'a')
        fid_log.write('Interpolation: Assigned main sequence durations of the stars (single stellar grid) \n')
        fid_log.close() 

    else:
        
        # Main sequence lifetimes [yrs]
        z = np.log10(np.array(MS_duration_grid_B))
        f = interpolate.RegularGridInterpolator((np.log10(m_grid_B),B_strength_grids), z.T)

        # Make the operation
        # It only works for stars within the interpolation ranges
        ind_m = (m>=np.min(m_grid_B))*(m<=np.max(m_grid_B))*(Binit>=np.min(B_strength_grids))*(Binit<=np.max(B_strength_grids))
        xy_new = list(zip(np.log10(m[ind_m]),Binit[ind_m]))
        t_MS_m = np.zeros(nbr_stars)
        t_MS_m[ind_m] = 10**np.array(f(xy_new))
        
        # Outside the mass range, let's just pick lowest and highest for now
        t_MS_m[m<np.min(m_grid_B)] = np.max(t_MS_m[ind_m])
        t_MS_m[m>np.max(m_grid_B)] = np.min(t_MS_m[ind_m])  
        
        # Tell the log
        fid_log = open(log_name,'a')
        fid_log.write('2D Interpolation: Assigned main-sequence durations of the stars (magnetic single stellar grid) \n')
        fid_log.close() 
        
    if compute_binaries:
        # Duration of the stripped phase [yrs]
        t_strip = 10**np.interp(np.log10(mstrip),np.log10(mstrip_grid[ind_full_strip]),
                                np.log10(strip_duration_grid[ind_full_strip]))
        # # # UPDATE HERE: instead of extrapolating, use realistic ages from pure helium star models. / Ylva 28 Feb 2022 
        t_strip = 10**extrapolate(np.log10(mstrip),np.log10(t_strip),
                                  np.log10(mstrip_grid[ind_full_strip]),np.log10(strip_duration_grid[ind_full_strip]))

        # Tell the log
        fid_log = open(log_name,'a')
        fid_log.write('Interpolation: Assigned durations of the stripped phases (binary stellar grid) \n')
        fid_log.close() 

        # Timescales for the primary in the binary systems
        birthday_m1 = birthday[primary]  # When they are born
        t_MS_m1 = t_MS_m[primary]        # How long their main sequence is
        t_lifetime_m1 = t_lifetime_m[primary]

        # Update the lifetimes of the stars that strip
        # For the stripped stars this is just the main sequence lifetime + the time as stripped
        lifetime_m_strip = t_MS_m1[ind_strip]+t_strip[ind_strip]
        tmp = t_lifetime_m[primary]
        t_lifetime_m1[ind_strip] = lifetime_m_strip
        t_lifetime_m[primary] = t_lifetime_m1

        # Tell the log
        fid_log = open(log_name,'a')
        fid_log.write('Updating the lifetimes of stars that get stripped to tau_MS + tau_strip (somewhat inaccurate for Case A) \n')
        fid_log.close() 



        # # # # # MASS ACCRETION EFFICIENCY, beta
        # 
        # Need to update the masses of the secondaries as well

        # Initiate an array with the masses of the secondaries, to begin with nothing has been changed
        m2_after_interaction = copy.copy(m2)

        # Assign masses of secondaries after interaction (interaction occurs once primary has reached HG so after t_MS_m)
        m2_after_interaction[ind_caseA] = m2_after_interaction[ind_caseA] + beta_MS*menv[ind_caseA]
        m2_after_interaction[ind_caseB] = m2_after_interaction[ind_caseB] + beta_HG*menv[ind_caseB]
        m2_after_interaction[ind_caseB_CEE] = m2_after_interaction[ind_caseB_CEE] + beta_HG_CEE*menv[ind_caseB_CEE]

        # Tell the log
        fid_log = open(log_name,'a')
        fid_log.write('Calculated the post-interaction masses of the secondaries with stripped stars. \n')
        fid_log.close() 


        # This array is for the primaries after interaction (if it occurred), i.e., stripped stars and mergers
        m1_after_interaction = copy.copy(m1)

        # Edit the masses of the primaries that became stripped stars
        m1_after_interaction[ind_strip] = mstrip[ind_strip]


        # Mergers: m1+m2, beta = 1
        m2_after_interaction[ind_merge] = 0.   # All mergers lose the secondary star
        m1_after_interaction[ind_merge] = m1[ind_merge] + m2[ind_merge]  # and the merger products are saved in the primary

        # Tell the log
        fid_log = open(log_name,'a')
        fid_log.write('Assuming that mergers are conservative and no mass is lost from the systems. -> dont trust the mergers too much \n')
        fid_log.close() 


        if iii == 0:
            if save_figs:
                # # # # Plot the M_before - M_after # # # # # # # #
                fig, (ax,ax2) = plt.subplots(1,2,figsize=(15,6))
                xlim = np.array([0,Minit_strip_max+1])

                # For the initial primary star
                ax.plot(m1,m1_after_interaction,'.',label='all')
                ax.plot(m1[ind_strip],m1_after_interaction[ind_strip],'.',label='stripped')
                ax.plot(m1[ind_merge],m1_after_interaction[ind_merge],'.',label='mergers')
                ax.legend(loc=0,fontsize=0.7*fsize)
                ax.set_xlim(xlim)
                ax.set_ylim([0,2*Minit_strip_max])
                ax.set_xlabel('Initial mass [$M_{\\odot}$]')
                ax.set_ylabel('Mass after interaction [$M_{\\odot}$]')
                ax.set_title('Mass of initial primary star')

                # For the secondary stars
                ax2.plot(m2,m2_after_interaction,'.',label='all')
                ax2.plot(m2[ind_strip],m2_after_interaction[ind_strip],'.',label='stripped')
                ax2.plot(m2[ind_merge],m2_after_interaction[ind_merge],'.',label='mergers')
                ax2.legend(loc=0,fontsize=0.7*fsize)
                ax2.set_xlim(xlim)
                ax2.set_ylim([0,2*Minit_strip_max])
                ax2.set_xlabel('Initial mass [$M_{\\odot}$]')
                ax2.set_ylabel('Mass after interaction [$M_{\\odot}$]')
                ax2.set_title('Mass of initial secondary star')

                fig.savefig(plots_dir+'/M_before_after.png',format='png',bbox_inches='tight',pad_inches=0.1)
                plt.close(fig)

                # I think the kink seen in the mergers at ~5 MSun is because the radius limit that has a little kink there.
                # # # # # # # # # # # # # # # # #

                # # # # Plot the M_strip - M2_after # # # # # # # #
                fig, ax = plt.subplots(1,1,figsize=(8,6))
                ind_tmp = m1>8.
                ax.plot(mstrip[ind_tmp*ind_caseA],m2_after_interaction[ind_tmp*ind_caseA],'.',alpha=0.5,label='Case A')
                ax.plot(mstrip[ind_tmp*ind_caseB],m2_after_interaction[ind_tmp*ind_caseB],'.',alpha=0.5,label='Case B')
                ax.plot(mstrip[ind_tmp*ind_caseB_CEE],m2_after_interaction[ind_tmp*ind_caseB_CEE],'.',alpha=0.5,label='CEE')

                ax.legend(loc=0,fontsize=0.7*fsize)

                ax.set_xlabel('stripped star mass [$M_{\\odot}$]')
                ax.set_ylabel('Companion star mass [$M_{\\odot}$]')
                ax.set_ylim([0,35])
                fig.savefig(plots_dir+'/Mstrip_M2.png',format='png',bbox_inches='tight',pad_inches=0.1)
                plt.close(fig)
                # # # # # # # # # # # # # # # # #

                # Tell the log
                fid_log = open(log_name,'a')
                fid_log.write('Saving a before-after diagram of masses\n')
                fid_log.write('Saving a Mstrip-M2 diagram\n\n')
                fid_log.close() 



        # # # # # PERIODS AFTER INTERACTION
        #
        # Below - lots of calculations -> check if you want, but don't erase. 
        # 
        # Non-conservative mass transfer.
        # 
        # See Onno Pols lecture notes on binaries for reaching the equation below:
        # 
        # $\dfrac{\dot{a}}{a} = -2 \dfrac{\dot{M}_d}{M_d} \left( 1 - \beta \dfrac{M_d}{M_a} - (1-\beta) (\gamma +\frac{1}{2})\dfrac{M_d}{M_d + M_a} \right)$
        # 
        # It can also be written as
        # 
        # $\dfrac{\dot{a}}{a} = -2\left( \dfrac{\dot{M}_d}{M_d} + \dfrac{\dot{M}_a}{M_a} - (\gamma +\dfrac{1}{2})\dfrac{\dot{M}}{M} \right)$
        # 
        # where $M = M_d + M_a$, $M_a$ is the mass of the accretor, and $M_d$ is the mass of the donor. This is because $\dot{M}_d = - \beta \dot{M}_a$ and therefore also $\dot{M} = \dot{M}_d + \dot{M}_a  = (1-\beta)\dot{M}_d$.
        # 
        # To begin with, we can remove the time dependence:
        # 
        # $\dfrac{\dot{a}}{a} = \dfrac{da}{a}/dt$
        # 
        # $-2\left( \dfrac{\dot{M}_d}{M_d} + \dfrac{\dot{M}_a}{M_a} - (\gamma +\dfrac{1}{2})\dfrac{\dot{M}}{M} \right) = -2\left( \dfrac{dM_d}{M_d} + \dfrac{dM_a}{M_a} -  (\gamma + \dfrac{1}{2})\dfrac{dM}{M} \right)/dt$
        # 
        # $\dfrac{da}{a} = -2\left( \dfrac{dM_d}{M_d} + \dfrac{dM_a}{M_a} -  (\gamma + \dfrac{1}{2})\dfrac{dM}{M} \right)$
        # 
        # To solve this, we start with the left-hand-side (LHS) of the equation, integrating it:
        # 
        # $\int_{a_i}^{a_f}\dfrac{da}{a} = \int_{\ln a_i}^{\ln  a_f} d\ln a = \ln a_f - \ln a_i = \ln  \left( \dfrac{a_f}{a_i} \right)$
        # 
        # The right hand side (RHS) is also integrated
        # 
        # $-2\left( \int_{M_{d,i}}^{M_{d,f}} \dfrac{dM_d}{M_d} + \int_{M_{a,i}}^{M_{a,f}} \dfrac{dM_a}{M_a} - \int_{M_i}^{M_f} (\gamma + \dfrac{1}{2})\dfrac{dM}{M} \right)$
        # 
        # ### Constant $\gamma$
        # Assuming that $\gamma$ is a constant gives
        # 
        # $-2\left( \int_{\ln M_{d,i}}^{\ln M_{d,f}} d\ln M_d + \int_{\ln M_{a,i}}^{\ln M_{a,f}} d\ln M_a - (\gamma + \dfrac{1}{2}) \int_{\ln M_i}^{\ln M_f} d\ln M \right)$
        # 
        # $-2\left(\ln M_{d,f} - \ln M_{d,i} + \ln M_{a,f} - \ln M_{a,i} - (\gamma + \dfrac{1}{2})(\ln M_f -  \ln M_i)\right)$
        # 
        # $-2 \left( \ln \dfrac{M_{d,f} M_{a,f}}{M_{d,i} M_{a,i}} - (\gamma + \dfrac{1}{2})\ln \dfrac{M_f}{M_i} \right)$
        # 
        # $\ln \left(\dfrac{M_{d,f} M_{a,f}}{M_{d,i} M_{a,i}}\right)^{-2} + \ln \left(\dfrac{M_{d,f} + M_{a,f}}{M_{d,i} + M_{a,i}}\right)^{2\gamma+1}$
        # 
        # $\ln \left( \left(\dfrac{M_{d,i} M_{a,i}}{M_{d,f} M_{a,f}}\right)^{2} \left(\dfrac{M_{d,f} + M_{a,f}}{M_{d,i} + M_{a,i}}\right)^{2\gamma+1}  \right)$
        # 
        # Putting the LHS and RHS together means that:
        # 
        # $\dfrac{a_f}{a_i} = \left(\dfrac{M_{d,i} M_{a,i}}{M_{d,f} M_{a,f}}\right)^{2} \left(\dfrac{M_{d,f} + M_{a,f}}{M_{d,i} + M_{a,i}}\right)^{2\gamma+1}$
        # 
        # ### Isotropic re-emission
        # In case the mass transfer efficiency is 1 ($\beta = 1$), the $dM$ term disappears, meaning that it is independent on whether $\gamma$ is a constant or not. Then 
        # 
        # $\dfrac{a_f}{a_i} =  \left( \dfrac{M_{d,i}M_{a,i}}{M_{d,f}M_{a,f}} \right)^2$
        # 
        # I case the mass transfer was completely non-conservative ($\beta = 0$), then $M_a$ is a constant. That means that we can account for isotropic re-emission and set $\gamma = M_d/M_a = (M - M_a)/M_a = M/M_a - 1$. This results in:
        # 
        # $\int _{M_i}^{M_f} (\gamma + \dfrac{1}{2}) \dfrac{dM}{M} = \int _{M_i}^{M_f} (\dfrac{M}{M_a} - \dfrac{1}{2}) \dfrac{dM}{M} = \dfrac{1}{M_a} \int_{M_i}^{M_f} dM - \dfrac{1}{2}\int_{\ln M_i}^{\ln M_f} d\ln M = \dfrac{M_f - M_i}{M_a} + \ln (M_i/M_f)^{1/2}$
        # 
        # which means that then 
        # 
        # $\ln \dfrac{a_f}{a_i} = \ln \left( \dfrac{M_{d,i}M_{a,i}}{M_{d,f}M_{a,f}} \right)^2 + 2\dfrac{M_f - M_i}{M_a} + \ln (M_i/M_f)$
        # 
        # 
        # ### Circumbinary ring
        # In the case of a circumbinary ring, the $\gamma = \dfrac{(M_d + M_a)^2}{M_d M_a} \sqrt{\dfrac{a_{\rm ring}}{a}}$. Following Artymowicz \& Lubow 1994, we set $a_{ring} = 2a$ and get $\gamma = \sqrt{2}\dfrac{(M_d + M_a)^2}{M_d M_a}$ (see also Zapartas et al. 2017a). 
        # 
        # For fully non-conservative mass transfer ($\beta = 0$), $M_a$ is constant. This means that the RHS becomes
        # 
        # $-2\left( \int_{M_{d,i}}^{M_{d,f}} \dfrac{dM_d}{M_d} + \int_{M_{a,i}}^{M_{a,f}} \dfrac{dM_a}{M_a} - \int_{M_i}^{M_f} (\gamma + \dfrac{1}{2})\dfrac{dM}{M} \right) = -2 \ln \left( \dfrac{M_{d,f}}{M_{d,i}} \right) + 2 \int _{M_i}^{M_f} \gamma \dfrac{dM}{M} + \int _{M_i}^{M_f} \dfrac{dM}{M} = -2 \ln \left( \dfrac{M_{d,f}}{M_{d,i}} \right) + \dfrac{2\sqrt{2}}{M_a} \int _{M_i}^{M_f} \dfrac{M}{(M-M_a)} dM + \ln \left( \dfrac{M_f}{M_i} \right) = -2 \ln \left( \dfrac{M_{d,f}}{M_{d,i}} \right) + \dfrac{2\sqrt{2}}{M_a} \left( M_a \ln \left( \dfrac{M_f - M_a}{M_i-M_a} \right) + M_f - M_i \right) + \ln \left( \dfrac{M_f}{M_i} \right) $
        # 
        # And can then be equaled with the left-hand side
        # 
        # For fully conservative mass transfer, the integral with the $\gamma$ disappears since $M$ is a constant. The separation is then calculated in the same way as for the other cases:
        # 
        # $\dfrac{a_f}{a_i} =  \left( \dfrac{M_{d,i}M_{a,i}}{M_{d,f}M_{a,f}} \right)^2$
        # 
        # 
        # ### Fast wind
        # Should I calculate this? Is this necessary? ($\gamma = M_a/M_d$)
        # 
        # ### Step from separation to period
        # 
        # And, finally, the separation $a$ can be translated to a period using Kepler III:
        # 
        # $\dfrac{P^2}{a^3} = \dfrac{4\pi}{G(M_d + M_a)}$
        # 
        # where the gravitational constant $G = 4\pi $ AU$^3 $ yr$^{-2} M_{\odot}^{-1}$ 

        # Tell the log
        fid_log = open(log_name,'a')
        fid_log.write('Calculating new periods following the treatment of angular momentum assumed during mass transfer.\n')
        fid_log.close() 


        # Calculate the period distribution after mass transfer (RLOF)

        # Isotropic re-emission
        # -- only works for beta = 0 or beta = 1 
        if angmom == 'isotropic_reemission':

            # Tell the log
            fid_log = open(log_name,'a')
            fid_log.write('Going to use isotropic re-emission...\n')
            fid_log.close() 

            # CASE A MASS TRANSFER
            # In case A mass transfer is fully conservative
            if beta_MS == 1.:

                # The final integral on the right side can be removed
                afinal[ind_caseA] = (ainit[ind_caseA]*
                         ((m1[ind_caseA]*m2[ind_caseA]/(mstrip[ind_caseA]*m2_after_interaction[ind_caseA]))**2.))

            # If case A mass transfer is fully non-conservative
            elif beta_MS == 0.:

                # Since this is isotropic re-emission I add the corresponding terms to the equation
                Mi = m1[ind_caseA] + m2[ind_caseA]
                Mf = mstrip[ind_caseA] + m2_after_interaction[ind_caseA]
                RHS = (np.log((m1[ind_caseA]*m2[ind_caseA]/(mstrip[ind_caseA]*m2_after_interaction[ind_caseA]))**2.) +
                      2.*(Mf-Mi)/m2_after_interaction[ind_caseA] + np.log(Mi/Mf))
                afinal[ind_caseA] = ainit[ind_caseA]*np.exp(RHS)

            else:
                fid_log = open(log_name,'a')
                fid_log.write('ERROR: isotropic re-emission currently only works for beta = 1 or 0!\n')
                fid_log.close() 


            # CASE B MASS TRANSFER
            # Same idea as for Case A mass transfer
            if beta_HG  == 1.:

                # The final integral on the right side can be removed
                afinal[ind_caseB] = (ainit[ind_caseB]*
                         ((m1[ind_caseB]*m2[ind_caseB]/(mstrip[ind_caseB]*m2_after_interaction[ind_caseB]))**2.))

            # If case B mass transfer is fully non-conservative
            elif  beta_HG == 0.:

                # Since this is isotropic re-emission I add the corresponding terms to the equation
                Mi = m1[ind_caseB] + m2[ind_caseB]
                Mf = mstrip[ind_caseB] + m2_after_interaction[ind_caseB]
                RHS = (np.log((m1[ind_caseB]*m2[ind_caseB]/(mstrip[ind_caseB]*m2_after_interaction[ind_caseB]))**2.) +
                      2.*(Mf-Mi)/m2_after_interaction[ind_caseB] + np.log(Mi/Mf))
                afinal[ind_caseB] = ainit[ind_caseB]*np.exp(RHS)

            else:
                fid_log = open(log_name,'a')
                fid_log.write('ERROR: isotropic re-emission currently only works for beta = 1 or 0!\n')
                fid_log.close() 


        # Circumbinary ring
        # -- only works for beta = 0 or beta = 1 
        elif angmom == 'circumbinary_ring':

            # Tell the log
            fid_log = open(log_name,'a')
            fid_log.write('Going to use circumbinary ring...\n')
            fid_log.close() 

            # CASE A MASS TRANSFER:
            # In case A mass transfer is fully conservative
            if beta_MS == 1.:

                # The final integral on the right side can be removed
                afinal[ind_caseA] = (ainit[ind_caseA]*
                         ((m1[ind_caseA]*m2[ind_caseA]/(mstrip[ind_caseA]*m2_after_interaction[ind_caseA]))**2.))

            # If case A mass transfer is fully non-conservative
            elif beta_MS == 0.:
                # The total masses before and after interaction
                Mi = m1[ind_caseA] + m2[ind_caseA]
                Mf = mstrip[ind_caseA] + m2[ind_caseA]
                # Construct the right-hand side
                aa = -2.*np.log(mstrip[ind_caseA]/m1[ind_caseA])
                bb = (2.*np.sqrt(2.)/m2[ind_caseA])*(m2[ind_caseA]*np.log((Mf-m2[ind_caseA])/(Mi-m2[ind_caseA])) + Mf - Mi)
                cc = np.log(Mf/Mi)
                RHS = aa + bb + cc

                # The separation after mass transfer (equal LHS and RHS)
                afinal[ind_caseA] = ainit[ind_caseA]*np.exp(RHS)

            else:
                fid_log = open(log_name,'a')
                fid_log.write('ERROR: circumbinary ring currently only works for beta = 1 or 0!\n')
                fid_log.close()         

            # CASE B MASS TRANSFER
            # In case B mass transfer is fully conservative
            if beta_HG  == 1.:

                # The final integral on the right side can be removed
                afinal[ind_caseB] = (ainit[ind_caseB]*
                         ((m1[ind_caseB]*m2[ind_caseB]/(mstrip[ind_caseB]*m2_after_interaction[ind_caseB]))**2.))    


            # In case B mass transfer is fully non-conservative
            elif beta_HG == 0.:
                # The total masses before and after interaction
                Mi = m1[ind_caseB] + m2[ind_caseB]
                Mf = mstrip[ind_caseB] + m2[ind_caseB]
                # Construct the right-hand side
                aa = -2.*np.log(mstrip[ind_caseB]/m1[ind_caseB])
                bb = (2.*np.sqrt(2.)/m2[ind_caseB])*(m2[ind_caseB]*np.log((Mf-m2[ind_caseB])/(Mi-m2[ind_caseB])) + Mf - Mi)
                cc = np.log(Mf/Mi)
                RHS = aa + bb + cc

                # The separation after mass transfer (equal LHS and RHS)
                afinal[ind_caseB] = ainit[ind_caseB]*np.exp(RHS)

            else:
                fid_log = open(log_name,'a')
                fid_log.write('ERROR: circumbinary ring currently only works for beta = 1 or 0!\n')
                fid_log.close()            


        # Constant gamma
        # -- works for all beta
        elif angmom == 'gamma_constant':

            # Tell the log
            fid_log = open(log_name,'a')
            fid_log.write('Going to use constant gamma...\n')
            fid_log.close() 

            # Separation in AU
            # Case A mass transfer 
            afinal[ind_caseA] = (ainit[ind_caseA]*
                         ((m1[ind_caseA]*m2[ind_caseA]/(mstrip[ind_caseA]*m2_after_interaction[ind_caseA]))**2.)* 
                         ((m2_after_interaction[ind_caseA]+mstrip[ind_caseA]/(m2[ind_caseA]+m1[ind_caseA]))**(2.*gamma_MS + 1.)))
            # Case B mass transfer
            afinal[ind_caseB] = (ainit[ind_caseB]*
                         ((m1[ind_caseB]*m2[ind_caseB]/(mstrip[ind_caseB]*m2_after_interaction[ind_caseB]))**2.)* 
                         ((m2_after_interaction[ind_caseB]+mstrip[ind_caseB]/(m2[ind_caseB]+m1[ind_caseB]))**(2.*gamma_HG + 1.)))




        # # # FINAL PERIOD # # #    
        # Calculate the periods after interaction from the separations (days)
        Pfinal[ind_caseA] = ((4.0*(np.pi**2.0)*
                                      (afinal[ind_caseA]**3.0)/(G*(mstrip[ind_caseA]+m2[ind_caseA])))**0.5)
        Pfinal[ind_caseB] = ((4.0*(np.pi**2.0)*
                                      (afinal[ind_caseB]**3.0)/(G*(mstrip[ind_caseB]+m2[ind_caseB])))**0.5)

        # Tell the log
        fid_log = open(log_name,'a')
        fid_log.write('Updated the periods of systems that go through Case A and Case B stable mass transfer.\n')
        fid_log.close() 


        # Now, it could be that some of the systems merged during actual stable mass transfer (circumbinary ring)
        # We therefore update the indices here for what star becomes a stripped star

        # Calculate the Roche-lobe radii for the stripped star and the companion (in AU)
        # Roche lobe radius for the stripped star
        q1 = mstrip[ind_strip]/m2_after_interaction[ind_strip]
        rL_strip = 0.49*(q1**(2./3.))/(0.69*(q1**(2./3.)) + np.log(1.+(q1**(1./3.))))
        RL_strip = rL_strip*afinal[ind_strip]   # AU
        # Roche lobe radius for the secondary (after interaction)
        q2 = m2_after_interaction[ind_strip]/mstrip[ind_strip]
        rL_2 = 0.49*(q2**(2./3.))/(0.69*(q2**(2./3.)) + np.log(1.+(q2**(1./3.))))
        RL_2 = rL_2*afinal[ind_strip]   # AU

        # Radii of the stripped stars -- Check if this is already done!!!
        rstrip = 10**np.interp(np.log10(mstrip[ind_strip]),np.log10(mstrip_grid),np.log10(rstrip_grid))
        rstrip = 10**extrapolate(np.log10(mstrip[ind_strip]),np.log10(rstrip),np.log10(mstrip_grid),np.log10(rstrip_grid))
        rstrip = rstrip*u.R_sun.to(u.AU)   # now in AU
        # Minimum radii of the secondaries (that might have accreted)
        rm2_acc_lim = np.interp(m2_after_interaction[ind_strip], m_grid, R_ZAMS_grid)
        rm2_acc_lim = (extrapolate(m2_after_interaction[ind_strip], rm2_acc_lim, m_grid, R_ZAMS_grid))*u.R_sun.to(u.AU)  # AU

        # For the systems that fit in the orbit, none of the stars fill the Roche lobe
        ind_fit_in_orbit = ((rm2_acc_lim < RL_2)*(rstrip < RL_strip)) > 0.

        # Now, I will remove the systems that don't fit in the orbit
        ind_caseA[ind_strip][ind_fit_in_orbit == False] = 0
        ind_caseB[ind_strip][ind_fit_in_orbit == False] = 0


        # Tell the log
        fid_log = open(log_name,'a')
        fid_log.write('Also thought about systems that tighten and removed the ones that did not fit in the orbit anymore.\n')
        fid_log.close() 

        if iii == 0:
            if save_figs:
                # # # # Period distribution - after interaction # # # # # # # #
                fig, ax = plt.subplots(1,1,figsize=(8,6))
                ind_tmp = m1>8.
                ax.hist(np.log10(Pfinal[ind_tmp*ind_caseA]),100,label='Case A',alpha=0.5)
                ax.hist(np.log10(Pfinal[ind_tmp*ind_caseB]),100,label='Case B',alpha=0.5)
                ax.hist(np.log10(Pfinal[ind_tmp*ind_caseB_CEE]),100,label='CEE',alpha=0.5)
                ax.set_xlabel('log P/days')
                ax.legend(loc=0,fontsize=0.7*fsize)
                fig.savefig(plots_dir+'/P_after.png',format='png',bbox_inches='tight',pad_inches=0.1)
                plt.close(fig)
                # # # # # # # # # # # # # # # # #

                # Tell the log
                fid_log = open(log_name,'a')
                fid_log.write('Saved a period distribution diagram for after interaction.\n\n')
                fid_log.close() 


        # # # # # REJUVENATION
        # 
        # From Tout et al. (1997) there is in Section 5.1 a treatment for rejuvenation. 
        # 
        # $t' = \dfrac{\mu}{\mu '}\dfrac{\tau_{\text{MS}} '}{\tau_{\text{MS}}} t$
        # 
        # where $t'$ is the apparent age of the star right after mass accretion, $t$ is the apparent age of the star if it wouldn't have been rejuvenated, $\tau_{\text{MS}}$ is the main sequence lifetime of the accretor prior to accretion, $\tau_{\text{MS}}'$ is the main sequence lifetime for a star with the initial mass that is the same of the accretor after mass accretion. The parameters $\mu$ are included when the accretor has a convective core and are then $\mu = M_2$ and $\mu ' = M_2  '$.
        # 
        # This means that a star that is rejuvenated lives $t-t'$ years in addition to the new assumed lifetime of the star.


        # # # ADD REJUVENATION # # # 

        # Assign a lifetime of the mergers
        t_lifetime_merger = 10**np.interp(np.log10(m1_after_interaction),
                                          np.log10(m_grid[ind_life]),np.log10(lifetime_grid[ind_life]))
        t_lifetime_merger = 10**extrapolate(np.log10(m1_after_interaction),np.log10(t_lifetime_merger),
                                            np.log10(m_grid[ind_life]),np.log10(lifetime_grid[ind_life]))
        t_MS_merger = 10**np.interp(np.log10(m1_after_interaction), np.log10(m_grid), np.log10(MS_duration_grid))
        t_MS_merger = 10**extrapolate(np.log10(m1_after_interaction), np.log10(t_MS_merger), np.log10(m_grid), np.log10(MS_duration_grid))


        # Assign a lifetime of the accretors after mass accretion
        t_lifetime_acc = np.zeros(len(m2))
        t_lifetime_acc[ind_merge==False] = 10**np.interp(np.log10(m2_after_interaction[ind_merge==False]),
                                       np.log10(m_grid[ind_life]),np.log10(lifetime_grid[ind_life]))
        t_lifetime_acc[ind_merge==False] = 10**extrapolate(np.log10(m2_after_interaction[ind_merge==False]),np.log10(t_lifetime_acc[ind_merge==False]),
                                         np.log10(m_grid[ind_life]),np.log10(lifetime_grid[ind_life]))
        # Main sequence duration of accretors (some were just secondaries without interaction so it's the same as before)
        t_MS_acc = np.zeros(len(m2))
        t_MS_acc[ind_merge==False] = 10**np.interp(np.log10(m2_after_interaction[ind_merge==False]), np.log10(m_grid), np.log10(MS_duration_grid))
        t_MS_acc[ind_merge==False] = 10**extrapolate(np.log10(m2_after_interaction[ind_merge==False]), np.log10(t_MS_acc[ind_merge==False]), np.log10(m_grid), np.log10(MS_duration_grid))

        # Tell the log
        fid_log = open(log_name,'a')
        fid_log.write('Interpolation: New lifetimes and main sequence durations of the accretors and mergers. \n')
        fid_log.close() 

        # Set current age and apparent age at the time of the mass accretion
        #   For the mergers
        age_event_m1 = copy.copy(t_MS_m1)
        apparent_age_after_event_m1 = copy.copy(age_event_m1)
        #   For the accretors
        age_event_m2 = copy.copy(t_MS_m1)
        apparent_age_after_event_m2 = copy.copy(age_event_m2)


        if rejuv_choice == 'Tout97':

            # Tell the log
            fid_log = open(log_name,'a')
            fid_log.write('Using the method from Tout et al. (1997) to simulate the rejuvenation of accretors. \n')
            fid_log.write('Concerning mergers: this only works for mergers that occurred during the main sequence.\n')
            fid_log.close() 

            # Implementing the rejuvenation method of Tout et al. (1997) (used also in the BSE, Hurley+00,02)
            # This rejuvenation is for mass gainers and not mergers
            ind_c = ((m2 < 0.3)+(m2 > 1.3))>0  # These are stars with convective insides, mu = m2, mu_p = m2_after_interaction
            #ind_r = ((m2 > 0.3)*(m2 < 1.3))    # These are stars with radiative insides, mu = mu_p
            mu = np.ones(len(m2))
            mu_p = np.ones(len(m2))
            mu[ind_c] = m2[ind_c]
            mu_p[ind_c] = m2_after_interaction[ind_c]

            # This is the extra time that the star lives because of the rejuvenation
            # Assuming that the interaction occurs when the primary has reached TAMS
            #rejuv_time_m2[ind_strip] = ((1-(mu[ind_strip]/mu_p[ind_strip])*
            #                            (t_lifetime_acc[ind_strip]/t_lifetime_m2[ind_strip]))*t_MS_m1[ind_strip])
            # For the accretors, only the secondaries that have or will have a stripped companion
            age_event_m2[ind_strip] = t_MS_m1[ind_strip]
            apparent_age_after_event_m2[ind_strip] = ((mu[ind_strip]/mu_p[ind_strip])*
                                                      (t_MS_acc[ind_strip]/t_MS_m2[ind_strip])*age_event_m2[ind_strip])

            # Tell the log
            fid_log = open(log_name,'a')
            fid_log.write('Rejuvenated the accretors (these are all main sequence stars at the stripping time). \n')
            fid_log.close()    

            # For the mergers, I only rejuvenate the MSMS mergers
            #print 'Only employing rejuvenation for the MSMS mergers'
            mu[ind_merge] = m1[ind_merge]
            mu_p[ind_merge] = m1_after_interaction[ind_merge]
            #rejuv_time_m1[ind_mergeA] = ((1-(mu[ind_mergeA]/mu_p[ind_mergeA])*
            #                             (t_lifetime_merger[ind_mergeA]/t_lifetime_m1[ind_mergeA]))*t_MS_m1[ind_mergeA])
            age_event_m1[ind_merge] = t_MS_m1[ind_merge]
            apparent_age_after_event_m1[ind_merge] = ((mu[ind_merge]/mu_p[ind_merge])*
                                                       (t_MS_merger[ind_merge]/t_MS_m1[ind_merge])*age_event_m1[ind_merge])

            # Tell the log
            fid_log = open(log_name,'a')
            fid_log.write('Rejuvenated the MSMS mergers \n')
            fid_log.close()   




        if iii == 0:
            if save_figs:
                # # # # Rejuvenation testing # # # # # # # #
                fig, (ax,ax2) = plt.subplots(1,2,figsize=(15,6))

                xlim = np.array([0,Minit_strip_max+1])

                rejuv_time_m1 = age_event_m1-apparent_age_after_event_m1
                rejuv_time_m2 = age_event_m2-apparent_age_after_event_m2

                ax.plot(m1_after_interaction,rejuv_time_m1,'.',label='all')
                ax.plot(m1_after_interaction[ind_strip],rejuv_time_m1[ind_strip],'.',label='stripped')
                ax.plot(m1_after_interaction[ind_merge],rejuv_time_m1[ind_merge],'.',label='mergers')
                ax.legend(loc=0,fontsize=0.7*fsize)
                ax.set_xlim(xlim)
                ax.set_xlabel('Mass after interaction [$M_{\\odot}$]')
                ax.set_ylabel('Rejuvenation time [yrs]')
                ax.set_title('Initial primary')
                ax.set_ylim([0,8e8])


                ax2.plot(m2_after_interaction,rejuv_time_m2,'.',label='all')
                ax2.plot(m2_after_interaction[ind_strip],rejuv_time_m2[ind_strip],'.',label='stripped')
                ax2.plot(m2_after_interaction[ind_merge],rejuv_time_m2[ind_merge],'.',label='mergers')
                ax2.legend(loc=0,fontsize=0.7*fsize)
                ax2.set_xlim(xlim)
                ax2.set_xlabel('Mass after interaction [$M_{\\odot}$]')
                ax2.set_ylabel('Rejuvenation time [yrs]')
                ax2.set_title('Initial secondary')
                ax2.set_ylim([0,8e8])

                fig.savefig(plots_dir+'/rejuvenation.png',format='png',bbox_inches='tight',pad_inches=0.1)
                plt.close(fig)
                # # # # # # # # # # # # # # # # #

                # Tell the log
                fid_log = open(log_name,'a')
                fid_log.write('Saved a figure for rejuvenation \n\n')
                fid_log.close() 



    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                                                                             #
    #     - Properties at a certain time                                          #
    #                                                                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    # 
    # Want to know the stellar parameters of stars at a certain time


    # Let's start ugly: if evaluation_time is just one number, use that one, otherwise make a for loop...
    for et in range(len(evaluation_time)):
    
        # # # # # APPARENT AGE # # # # #
        # My plan here is to assign stellar properties for each star at a given time - the evaluation time (eval_time) 
        # Whether the star is present is determined by the presence of the primary. I neglect possible secondaries that are still
        # alive after the primary exploded. 

        # Give a time to evaluate the population at
        eval_time = copy.copy(evaluation_time[et]) 

        # Tell the log
        fid_log = open(log_name,'a')
        fid_log.write('Starting to evaluate population after '+str(eval_time)+' years.\n')
        fid_log.close() 


        # I will introduce arrays with the apparent age
        apparent_age_m = np.zeros(len(m))
        if compute_binaries:
            apparent_age_m1 = np.zeros(len(m1))
            apparent_age_m2 = np.zeros(len(m2))


        # (1) Assign apparent ages for the single stars (from birth, but also when they are compact objects)
        apparent_age_m[single] = eval_time-birthday[single]
        apparent_age_m[apparent_age_m < 0.] = 0.


        if compute_binaries:
            # (2) The systems that have not yet interacted or we assume are outside of the interaction mass range - actual ages
            # Stars that might interact in the future:
            ind_tmp1 = ((birthday_m1 < eval_time)*((birthday_m1+t_MS_m1) > eval_time)*ind_interaction_mass)
            # Stars that we assume do not interact, but are in a binary:
            ind_no_interaction = ind_interaction_mass == False
            ind_tmp2 = ((birthday_m1 < eval_time)*ind_no_interaction)   # allow them to be also compact objects
            ind_nyi = (ind_tmp1+ind_tmp2)>0
            apparent_age_m1[ind_nyi] = eval_time-birthday_m1[ind_nyi]
            apparent_age_m2[ind_nyi] = eval_time-birthday_m2[ind_nyi]


            # (3) Binaries that are too wide for interaction to occur even though they are in the right mass range
            # (allow for compact objects to have an age too)
            ind_wide = ((birthday_m1 < eval_time)*ind_interaction_mass*(ind_strip==False)*(ind_merge==False))
            apparent_age_m1[ind_wide] = eval_time-birthday_m1[ind_wide]
            apparent_age_m2[ind_wide] = eval_time-birthday_m2[ind_wide]


            # (4) The stars that have a stripped component 
            # (or single accretors, accretors with a compact companion or single stripped stars or stripped stars with CO)
            # a) Start with the stripped star (the lifetime is updated from before), these may sometimes be single or with compact companion
            ind_ns = (((birthday_m1+t_MS_m1) < eval_time)*((birthday_m1+t_lifetime_m1) > eval_time)*ind_strip)
            apparent_age_m1[ind_ns] = eval_time-birthday_m1[ind_ns]
            # b) The secondary also has an apparent age, this can be different from the stripped star 
            # (include also single secondaries and secondaries with compact object companions)
            ind_cs = (((birthday_m2+t_MS_m1) < eval_time)*
                      ((birthday_m2+age_event_m2-apparent_age_after_event_m2+t_lifetime_acc) > eval_time)*ind_strip)
            apparent_age_m2[ind_cs] = eval_time-birthday_m2[ind_cs]-(age_event_m2[ind_cs]-apparent_age_after_event_m2[ind_cs])


            # (5) The merger products
            ind_mn = ((birthday_m1+t_MS_m1 < eval_time)*
                      ((birthday_m1+age_event_m1-apparent_age_after_event_m1+t_lifetime_merger) > eval_time)*ind_merge)
            apparent_age_m1[ind_mn] = eval_time-birthday_m1[ind_mn]-(age_event_m1[ind_mn]-apparent_age_after_event_m1[ind_mn])
            apparent_age_m2[ind_mn] = 0.  # The secondaries should be gone


            # (6) The compact objects after stripped star systems
            # The evolved stripped stars themselves
            ind_sco = ind_strip*((birthday_m1+t_lifetime_m1) < eval_time)
            apparent_age_m1[ind_sco] = eval_time-birthday_m1[ind_sco]
            # The evolved companions
            ind_cco = ind_strip*((birthday_m2+age_event_m2-apparent_age_after_event_m2+t_lifetime_acc) < eval_time)
            apparent_age_m2[ind_cco] = eval_time-birthday_m2[ind_cco]-(age_event_m2[ind_cco]-apparent_age_after_event_m2[ind_cco])


            # Update the main apparent age array
            apparent_age_m[primary] = apparent_age_m1


            # THE STARS THAT ARE PRESENT AT THAT TIME
            # Choosing just the stars within the interaction range. 
            stars_present = (apparent_age_m != 0)*(m <= Minit_strip_max)*(m >= Minit_strip_min)
            stars_present_m1 = (apparent_age_m1 != 0)*ind_interaction_mass
            stars_present_m2 = (apparent_age_m2 != 0)*ind_interaction_mass

        else:

            # If there is no binaries, just pick the stars that are born and haven't died.
            #stars_present = (apparent_age_m != 0)*(eval_time < (birthday_m+t_lifetime_m))
            # I will allow for them to be compact objects
            stars_present = apparent_age_m != 0

        # Tell the log
        fid_log = open(log_name,'a')
        fid_log.write('Assigned apparent ages of the stars that are present in the population. \n\n')
        fid_log.close() 


        # # # # # EVOLUTIONARY STAGE # # # # #
        # This is for checking in which evolutionary stage the stars are in at the evaluation time. 
        # When envelope stripping occurs, the secondary is always a main sequence star, but it might evolve
        # into a giant when the companion is still a stripped star. The secondary might also explode and leave
        # a compact remnant during the stripped phase. 


        # Index array with the different possibilities (for the binaries)
        present_state = np.zeros(len(m))
        if compute_binaries:
            present_state_m1 = np.zeros(len(m1))
            present_state_m2 = np.zeros(len(m1))

        # 1 - main sequence star
        # 2 - stripped star
        # 3 - post main sequence star 
        # 4 - compact remnant (white dwarf, neutron star, black hole), not necessarily attached to the binary
        # 51 - merger product that is on the main sequence
        # 53 - merger product that is on the post main sequence
        # 54 - merger product that has become a compact object


        # (1) Single stars that are present
        #     (ignoring secondaries that are now single)
        # a) the ones that are on the main sequence
        ind_sms = (birthday < eval_time)*((birthday+t_MS_m) > eval_time)*single    # sms - "single, main sequence"
        present_state[ind_sms] = 1
        # b) the ones that are post the main sequence
        ind_spms = ((birthday+t_MS_m) < eval_time)*((birthday+t_lifetime_m) > eval_time)*single   # spms - "single, post main sequence"
        present_state[ind_spms] = 3
        # c) the ones that have become white dwarfs or exploded
        ind_sco = ((birthday+t_lifetime_m) < eval_time)*single    # sco - "single, compact object"
        present_state[ind_sco] = 4


        if compute_binaries:
            # (2) Main sequence primaries (these cannot have interacted yet, per definition)
            ind_m1ms = (birthday_m1 < eval_time)*((birthday_m1+t_MS_m1) > eval_time)    # m1ms - "M1 is on main sequence"
            present_state_m1[ind_m1ms] = 1
            # Since the secondary is less massive and interaction has not occurred, it is also a main sequence star
            present_state_m2[ind_m1ms] = 1



            # (3) Stripped star primaries
            ind_m1s = ((birthday_m1+t_MS_m1) < eval_time)*((birthday_m1+t_lifetime_m1) > eval_time)*ind_strip   # m1s - "M1 is stripped"
            present_state_m1[ind_m1s] = 2
            # 3a) such systems with main sequence secondaries
            ind_m2sms = ind_m1s*((birthday_m2+age_event_m2-apparent_age_after_event_m2+t_MS_acc) > eval_time)   # m2sms - "M2 with a stripped companion, but is on the main sequence"
            present_state_m2[ind_m2sms] = 1
            # 3b) such sysems with post-MS secondaries
            ind_m2spms = (ind_m1s*((birthday_m2+age_event_m2-apparent_age_after_event_m2+t_MS_acc) < eval_time)*
                          ((birthday_m2+age_event_m2-apparent_age_after_event_m2+t_lifetime_acc) > eval_time))    # m2spms - "M2 with a stripped companion, but is on the post main sequence"
            present_state_m2[ind_m2spms] = 3
            # single (runaway?) stripped stars or stripped stars with CO companion (secondary exploded)
            ind_m2sco = ind_m1s*((birthday_m2+age_event_m2-apparent_age_after_event_m2+t_lifetime_acc) < eval_time)   # m2sco - "M2 with a stripped companion, but has evolved to a compact object"  (usually these are white dwarfs, but maybe in an updated version of the code, these might occur also for higher mass stars)
            present_state_m2[ind_m2sco] = 4



            # (4) Post-main sequence primaries (no interaction or late interaction)
            ind_m1pms = (((birthday_m1+t_MS_m1) < eval_time)*((birthday_m1+t_lifetime_m1) > eval_time)*
                         (ind_merge == False)*(ind_strip == False))    # m1pms - "M1 is on the post main sequence"
            present_state_m1[ind_m1pms] = 3
            # 4a) such systems with main sequence secondaries
            ind_m2pmsms = ind_m1pms*(birthday_m2 < eval_time)*((birthday_m2+t_MS_m2) > eval_time)
            present_state_m2[ind_m2pmsms] = 1     # m2pmsms - "M2 with post main sequence companion while it is on main sequence"
            # 4b) such systems with post-MS secondaries
            ind_m2pmspms = ind_m1pms*((birthday_m2+t_MS_m2) < eval_time)*((birthday_m2+t_lifetime_m2) > eval_time)
            present_state_m2[ind_m2pmspms] = 3      # m2pmspms - "M2 with post main sequence companion while it also is on the post main sequence"
            # not possible to have single primaries from these types of systems



            # (5) Merger products 
            #    (these have special numbering to highlight that they merged)
            # 5a) main sequence merger products
            ind_m1mms = (((birthday_m1+t_MS_m1) < eval_time)*
                         ((birthday_m1+age_event_m1-apparent_age_after_event_m1+t_MS_merger) > eval_time)*ind_merge)
            present_state_m1[ind_m1mms] = 51
            present_state_m2[ind_m1mms] = 0
            # 5b) post-MS merger products
            ind_m1mpms = (((birthday_m1+age_event_m1-apparent_age_after_event_m1+t_MS_merger) < eval_time)*
                          ((birthday_m1+age_event_m1-apparent_age_after_event_m1+t_lifetime_merger) > eval_time)*ind_merge)
            present_state_m1[ind_m1mpms] = 53
            present_state_m2[ind_m1mpms] = 0
            # 5c) merger product is a compact object
            ind_m1mco = (((birthday_m1+age_event_m1-apparent_age_after_event_m1+t_lifetime_merger) < eval_time)*ind_merge)
            present_state_m1[ind_m1mco] = 54
            present_state_m2[ind_m1mco] = 0



            # (6) Primary is dead, didn't merge with secondary
            ind_m1co = ((birthday_m1 + t_lifetime_m1) < eval_time)*(ind_merge==False)
            present_state_m1[ind_m1co] = 4
            # 6a) secondary is a main sequence star
            ind_m2coms = (ind_m1co*(birthday_m2 < eval_time)*
                          ((birthday_m2+age_event_m2-apparent_age_after_event_m2+t_MS_acc) > eval_time))
            present_state_m2[ind_m2coms] = 1
            # 6b) secondary is a post main sequence star
            ind_m2copms = (ind_m1co*((birthday_m2+age_event_m2-apparent_age_after_event_m2+t_MS_acc) < eval_time)*
                           ((birthday_m2+age_event_m2-apparent_age_after_event_m2+t_lifetime_acc) > eval_time))
            present_state_m2[ind_m2copms] = 3
            # 6c) secondary is also dead
            ind_m2coco = ind_m1co*((birthday_m2+age_event_m2-apparent_age_after_event_m2+t_lifetime_acc) < eval_time)
            present_state_m2[ind_m2coco] = 4


            # (7) Late interaction objects, case C mass transfer and mergers
            #print 'Consider to account for late interaction phases somehow'


            # Update the general present_state array (including single stars too)
            present_state[primary] = present_state_m1

        # Tell the log
        fid_log = open(log_name,'a')
        fid_log.write('Checked the evolutionary states of the stars at the evaluation time. \n')
        fid_log.write('I have neglected late interaction objects. \n\n')
        fid_log.close() 


        #_________________________________________________
        # 
        # # # # # INTERPOLATION OF MESA MODELS # # # # #
        #_________________________________________________
        # 
        # Stellar properties from MESA models 
        # Interpolation scheme

        # Start with the stellar evolutionary models
        # Get the properties at scaled times over the main sequence, store them in matrices

        # # # # # Prepare the interpolation


        # The scaled times. (This will be fraction of time from ZAMS to TAMS for example)
        num_t  = 1000
        t_scaled = np.linspace(0,1,num_t)

        # If this is not for magnetic stars
        if compute_magnetic == False:

            # Storage for parameters
            # Main sequence
            duration_MS_sin = np.zeros(nbr_sims)
            MESA_params_MS_scaled = np.zeros([nbr_MESA_param,nbr_sims,num_t])

            # Post Main sequence
            reach_he_depletion_sin = [False]*nbr_sims
            duration_pMS_sin = np.zeros(nbr_sims)
            MESA_params_pMS_scaled = np.zeros([nbr_MESA_param,nbr_sims,num_t])

            if compute_binaries:
                # Stripped stars
                reach_he_depletion_strip = [False]*nbr_sims_bin
                duration_strip = np.zeros(nbr_sims_bin)
                MESA_params_strip_scaled = np.zeros([nbr_MESA_param_b,nbr_sims_bin,num_t])

            # Loop over the models in the single star grid
            for i in range(nbr_sims):

                # Get some properties that will help here
                center_h1 = MESA_params[i][col.index('center_h1')]
                center_he4 = MESA_params[i][col.index('center_he4')]
                star_age = MESA_params[i][col.index('star_age')]

                # Check whether the star finished the main sequence
                #if center_h1[-1] < 0.001:   # TEMPORARY: relaxing this condition temporarily because one of Zsolts models didn't fulfil this
                if center_h1[-1] < 0.01:

                    # # # # # # # # # # # # # # # # #
                    # # # #  MAIN SEQUENCE    # # # # 
                    # # # # # # # # # # # # # # # # #
                    # Get the main sequence duration
                    indices = np.arange(len(star_age))
                    ind_MS_sin_tmp = indices[ind_ZAMS[i]:ind_TAMS[i]]
                    duration_MS_sin[i] = star_age[ind_TAMS[i]]-star_age[ind_ZAMS[i]]
                    # This is the scaled times for the actual output
                    scaled_times_MS_sin_tmp = star_age[ind_MS_sin_tmp]/duration_MS_sin[i] 

                    # Now I will interpolate the properties at the different pre-defined scaled times
                    for cc in range(nbr_MESA_param):
                        MESA_params_MS_scaled[cc,i,:] = np.interp(t_scaled, scaled_times_MS_sin_tmp, MESA_params[i][cc][ind_MS_sin_tmp])

                    # # # # # # # # # # # # # # # # #
                    # # # #  POST MAIN SEQUENCE # # # 
                    # # # # # # # # # # # # # # # # #        
                    # I am going to just use the models that finished central helium burning,
                    # and also just until central helium burning, meaning that I remove the final, short-lasting phases.
                    # The helium limit needs to be a little larger because of the formation of helium just before explosion
                    if center_he4[-1] < 0.01:

                        reach_he_depletion_sin[i] = True

                        # Get the post-MS duration (removing after central helium depletion)
                        ind_he = indices[center_he4 < 0.01][0]
                        ind_pMS_sin_tmp = indices[ind_TAMS[i]:ind_he]
                        duration_pMS_sin[i] = star_age[ind_he]-star_age[ind_TAMS[i]]
                        # Scaled times for the actual output
                        scaled_times_pMS_sin_tmp = (star_age[ind_pMS_sin_tmp]-star_age[ind_TAMS[i]])/duration_pMS_sin[i]

                        # Now, interpolate the properties at the different pre-defined scaled times
                        for cc in range(nbr_MESA_param):
                            MESA_params_pMS_scaled[cc,i,:] = np.interp(t_scaled, scaled_times_pMS_sin_tmp, MESA_params[i][cc][ind_pMS_sin_tmp])


            # --- UPDATE HERE IF IT WORKS FOR SINGLE STARS ---
            if compute_binaries:
                # Loop over the stripped star evolutionary models 
                for i in range(nbr_sims_bin):

                    # Some parameters that are needed
                    center_h1_b = MESA_params_b[i][col_bin.index('center_h1')]
                    center_he4_b = MESA_params_b[i][col_bin.index('center_he4')]
                    RLOF_b = MESA_params_b[i][col_bin.index('rl_relative_overflow_1')]
                    log_abs_mdot_b = MESA_params_b[i][col_bin.index('log_abs_mdot')]
                    star_age_b = MESA_params_b[i][col_bin.index('star_age')]


                    indices = np.arange(len(center_he4_b))
                    # Cutting the track once mass transfer is completed
                    ind_tmp = RLOF_b*(center_he4_b > 0.5)
                    if m_bin_grid[i] < 4.:
                        ind_tmp = ((RLOF_b+(log_abs_mdot_b > -9))>0)*(center_he4_b > 0.5)
                    ind_finish_RLOF = indices[ind_tmp][-1]

                    # For the ones which reach helium depletion -- best case
                    if (center_h1_b[-1] < 0.001) and (center_he4_b[-1] < 0.01):

                        reach_he_depletion_strip[i] = True

                        # Get the duration of the stripped phase
                        ind_he = indices[center_he4_b < 0.01][0]
                        ind_strip_bin_tmp = indices[ind_finish_RLOF:ind_he]
                        duration_strip[i] = star_age_b[ind_he]-star_age_b[ind_finish_RLOF]

                    # But I still want to use the other models, so I will come up with a solution for that
                    else:

                        # Add an extra number in the index array
                        ind_strip_bin_tmp = np.concatenate([indices[ind_finish_RLOF:],[len(star_age_b)-1]])
                        # Get the duration of the stripped phase by interpolating the models that reached central helium depletion
                        duration_strip[i] = 10**np.interp(np.log10(mstrip_grid[i]),np.log10(mstrip_grid[ind_full_strip]),
                                        np.log10(strip_duration_grid[ind_full_strip]))
                        duration_strip[i] = 10**extrapolate(np.log10(mstrip_grid[i]),np.log10(duration_strip[i]),
                                           np.log10(mstrip_grid[ind_full_strip]),np.log10(strip_duration_grid[ind_full_strip]))
                        # Time for the end of central helium burning for this star
                        tendHe = star_age_b[ind_finish_RLOF]+duration_strip[i]

                        # Update the arrays (pick the last value)
                        for cc in range(nbr_MESA_param_b):
                            tmp_param = np.concatenate([MESA_params_b[i][cc],[MESA_params_b[i][cc][-1]]])
                            tmp_mat = copy.copy(MESA_params_b[i])
                            tmp_mat[cc] = copy.copy(tmp_param)
                            MESA_params_b[i] = tmp_mat
                            del tmp_param

                        # Update the time array (make sure it uses tendHe)
                        star_age_b = np.concatenate([star_age_b,[tendHe]])
                        tmp = copy.copy(MESA_params_b[i])
                        tmp[col_bin.index('star_age')] = copy.copy(star_age_b)
                        MESA_params_b[i] = tmp

                    # Calculate the scaled times for the output
                    scaled_times_strip_tmp = (star_age_b[ind_strip_bin_tmp]-star_age_b[ind_finish_RLOF])/duration_strip[i]

                    # Interpolate the properties at the different pre-defined scaled times
                    for cc in range(nbr_MESA_param_b):
                        MESA_params_strip_scaled[cc,i,:] = np.interp(t_scaled, scaled_times_strip_tmp, MESA_params_b[i][cc][ind_strip_bin_tmp])

                 # UPDATE HERE: Add here a loop for the pure helium star models and a mass limit / Ylva 28 Feb 2022 

        # In case indeed there are magnetic stars
        else:

            # Storage for parameters
            # Main sequence
            duration_MS_sin = np.zeros([nbr_B_grids,nbr_B_sims[0]])  # I am thinking rows for B-spread, columns for mass #np.zeros(nbr_sims)
            MESA_params_MS_scaled = np.zeros([nbr_MESA_param,nbr_B_grids,nbr_B_sims[0],num_t])

            # Post Main sequence
            reach_he_depletion_sin = np.zeros([nbr_B_grids,nbr_B_sims[0]])   #[False]*nbr_sims
            duration_pMS_sin = np.zeros([nbr_B_grids,nbr_B_sims[0]]) #np.zeros(nbr_sims)
            MESA_params_pMS_scaled = np.zeros([nbr_MESA_param,nbr_B_grids,nbr_B_sims[0],num_t])


            # There is not yet any binary possibilities for the magnetic stars
            #if compute_binaries:
            #    # Stripped stars
            #    reach_he_depletion_strip = [False]*nbr_sims_bin
            #    duration_strip = np.zeros(nbr_sims_bin)
            #    MESA_params_strip_scaled = np.zeros([nbr_MESA_param_bin,nbr_sims_bin,num_t])

            # Loop over magnetic field
            for j in range(nbr_B_grids):
                # Loop over the models in the single star grid
                for i in range(nbr_B_sims[j]):    # They should all be the same number in nbr_B_sims[]

                    # Get some properties that will help here
                    center_h1 = MESA_params_B[j][i][col.index('center_h1')]
                    center_he4 = MESA_params_B[j][i][col.index('center_he4')]
                    star_age = MESA_params_B[j][i][col.index('star_age')]

                    # Check whether the star finished the main sequence
                    #if center_h1[-1] < 0.001:   # TEMPORARY: relaxing this condition temporarily because one of Zsolts models didn't fulfil this
                    if center_h1[-1] < 0.01:

                        # # # # # # # # # # # # # # # # #
                        # # # #  MAIN SEQUENCE    # # # # 
                        # # # # # # # # # # # # # # # # #
                        # Get the main sequence duration
                        indices = np.arange(len(star_age))
                        ind_MS_sin_tmp = indices[ind_ZAMS_B[j][i]:ind_TAMS_B[j][i]]
                        duration_MS_sin[j,i] = star_age[ind_TAMS_B[j][i]]-star_age[ind_ZAMS_B[j][i]]
                        # This is the scaled times for the actual output
                        scaled_times_MS_sin_tmp = star_age[ind_MS_sin_tmp]/duration_MS_sin[j,i] 

                        # Now I will interpolate the properties at the different pre-defined scaled times
                        for cc in range(nbr_MESA_param):
                            MESA_params_MS_scaled[cc,j,i,:] = np.interp(t_scaled, scaled_times_MS_sin_tmp, MESA_params_B[j][i][cc][ind_MS_sin_tmp])

                        # # # # # # # # # # # # # # # # #
                        # # # #  POST MAIN SEQUENCE # # # 
                        # # # # # # # # # # # # # # # # #        
                        # I am going to just use the models that finished central helium burning,
                        # and also just until central helium burning, meaning that I remove the final, short-lasting phases.
                        # The helium limit needs to be a little larger because of the formation of helium just before explosion
                        if center_he4[-1] < 0.01:

                            reach_he_depletion_sin[j,i] = True

                            # Get the post-MS duration (removing after central helium depletion)
                            ind_he = indices[center_he4 < 0.01][0]
                            ind_pMS_sin_tmp = indices[ind_TAMS_B[j][i]:ind_he]
                            duration_pMS_sin[j,i] = star_age[ind_he]-star_age[ind_TAMS[i]]
                            # Scaled times for the actual output
                            scaled_times_pMS_sin_tmp = (star_age[ind_pMS_sin_tmp]-star_age[ind_TAMS_B[j][i]])/duration_pMS_sin[j,i]

                            # Now, interpolate the properties at the different pre-defined scaled times
                            for cc in range(nbr_MESA_param):
                                MESA_params_pMS_scaled[cc,j,i,:] = np.interp(t_scaled, scaled_times_pMS_sin_tmp, MESA_params_B[j][i][cc][ind_pMS_sin_tmp])


        # Tell the log
        fid_log = open(log_name,'a')
        fid_log.write('Interpolation prepared for main sequence, post-main sequence, and stripped stars (if included) \n')
        fid_log.write('Allowing for extrapolation at the edges if the model did not reach central helium depletion.\n')
        fid_log.write('(Just assuming the star has the same properties until when central helium depletion is expected.) \n')
        fid_log.close() 

        if iii == 0:
            if save_figs:
                # # # #  Verification plots

                if compute_binaries:
                    # Just checking so that the cutting of the RLOF works well (the colored lines are what I will use.)
                    ww=6
                    hh=8
                    fig, ax = plt.subplots(1,1, figsize=(ww,hh))
                    for i in range(nbr_sims_bin):

                        log_Teff_b = MESA_params_b[i][col_bin.index('log_Teff')]
                        log_L_b = MESA_params_b[i][col_bin.index('log_L')]
                        center_he4_b = MESA_params_b[i][col_bin.index('center_he4')]
                        RLOF_b = MESA_params_b[i][col_bin.index('rl_relative_overflow_1')]
                        log_abs_mdot_b = MESA_params_b[i][col_bin.index('log_abs_mdot')]

                        ax.plot(log_Teff_b,log_L_b,'-',color=[.7]*3)

                        indices = np.arange(len(log_Teff_b))
                        ind_tmp = RLOF_b*(center_he4_b > 0.5)
                        if m_bin_grid[i] < 4.:
                            ind_tmp = ((RLOF_b+(log_abs_mdot_b > -9))>0)*(center_he4_b > 0.5)
                        ind_finish_RLOF = indices[ind_tmp][-1]
                        if reach_he_depletion_strip[i]:
                            ind_he = indices[center_he4_b < 0.01][0]
                            ind_strip_bin_tmp = indices[ind_finish_RLOF:ind_he]
                            ax.plot(log_Teff_b[ind_strip_bin_tmp],log_L_b[ind_strip_bin_tmp],lw=3)
                        else:
                            ax.plot(log_Teff_b[ind_finish_RLOF:],log_L_b[ind_finish_RLOF:],lw=3)    
                    ax.invert_xaxis()
                    ax.set_xlabel('log Teff')
                    ax.set_ylabel('log L')
                    fig.savefig(plots_dir+'/MESA_interpolation_RLOFcut.png',format='png',bbox_inches='tight',pad_inches=0.1)
                    plt.close(fig)


                # Save a grid of plots for each B-grid to check the interpolation doesn't look crazy. 
                if compute_magnetic:
                    # Make a plot for each parameter
                    cmap = matplotlib.cm.get_cmap('plasma')
                    for j in range(nbr_B_grids):
                        CC = 5
                        RR = int(np.ceil(len(col)/np.float_(CC)))
                        fig, axes = plt.subplots(RR,CC,figsize=(15,15))
                        c = 0
                        r = 0
                        for cc in range(len(col)):
                            clrs = cmap(np.linspace(0.1,1.,nbr_B_sims[j]))
                            for i in range(nbr_B_sims[j]):
                                axes[r,c].plot(t_scaled,MESA_params_MS_scaled[cc,j,i,:],'-',color=clrs[i])
                            axes[r,c].set_xlabel('Scaled time')
                            axes[r,c].set_ylabel(col[cc].replace('_','-'))
                            if (r==0) and (c==0):
                                axes[r,c].set_title('Binit='+B_grids[j]+'G')
                            c=c+1
                            if (c > (CC-1)):
                                r=r+1
                                c=0
                        fig.savefig(plots_dir+'/MESA_interpolation_'+B_grids[j]+'G.png',format='png',bbox_inches='tight',pad_inches=0.1)
                        plt.close(fig)


                # Main-sequence interpolation verification plots
                else:
                    # Make a plot for each parameter
                    cmap = matplotlib.cm.get_cmap('plasma')
                    CC = 5
                    RR = int(np.ceil(len(col)/np.float_(CC)))
                    fig, axes = plt.subplots(RR,CC,figsize=(15,15))
                    c = 0
                    r = 0
                    for cc in range(len(col)):
                        clrs = cmap(np.linspace(0.1,1.,nbr_sims))
                        for i in range(nbr_sims):
                            axes[r,c].plot(t_scaled,MESA_params_MS_scaled[cc,i,:],'-',color=clrs[i])
                        axes[r,c].set_xlabel('Scaled time')
                        axes[r,c].set_ylabel(col[cc].replace('_','-'))
                        if (r==0) and (c==0):
                            axes[r,c].set_title('Main-sequence star interpolation')
                        c=c+1
                        if (c > (CC-1)):
                            r=r+1
                            c=0
                    fig.savefig(plots_dir+'/MESA_interpolation_MS.png',format='png',bbox_inches='tight',pad_inches=0.1)
                    plt.close(fig)




        # # # # Use the interpolation of the MESA models

        # Storage space for properties of all stars
        # # Primaries and single stars
        MESA_params_interp = np.zeros([nbr_MESA_param,nbr_stars])

        # --- UPDATE HERE IF SINGLES WORK ---
        if compute_binaries:
            # # Secondaries
            MESA_params_interp_m2 = np.zeros([nbr_MESA_param,sum(primary)])


        # Start with the main sequence stars

        # # # MAIN SEQUENCE: primaries and singles # # # 
        # Get the main sequence primaries and single stars that are in the population at this time
        ind_MS_present = stars_present*(((present_state == 1)+(present_state == 51)) > 0)

        # Calculate their scaled ages 
        duration_MS = copy.copy(t_MS_m)
        if compute_binaries:
            tmp = duration_MS[primary]
            tmp[present_state_m1 == 51] = t_MS_merger[present_state_m1 == 51]     # mergers
            duration_MS[primary] = tmp
        actual_scaled_times_MS = apparent_age_m[ind_MS_present]/duration_MS[ind_MS_present]

        if compute_binaries:
            # # # MAIN SEQUENCE: secondaries # # #
            # Get which secondary stars are on the main sequence
            ind_MS_present_m2 = stars_present_m2*(present_state_m2 == 1)

            # Calculate their scaled ages
            # the secondaries that have not changed because of interaction
            duration_MS_m2 = copy.copy(t_MS_m2)
            # the accretors
            duration_MS_m2[(present_state_m2==1)*(present_state_m1==2)] = t_MS_acc[(present_state_m2==1)*(present_state_m1==2)]
            actual_scaled_times_MS_m2 = apparent_age_m2[ind_MS_present_m2]/duration_MS_m2[ind_MS_present_m2]


        # # # POST MAIN SEQUENCE: primaries and singles # # #
        # Pick the regular post main sequence stars and the merger post main sequence stars
        ind_pMS_present = stars_present*(((present_state == 3)+(present_state == 53)) > 0)

        # Calculate their scaled ages
        duration_pMS = t_lifetime_m-t_MS_m
        if compute_binaries:
            tmp = duration_pMS[primary]
            tmp[present_state_m1 == 53] = t_lifetime_merger[present_state_m1 == 53] - t_MS_merger[present_state_m1 == 53]
            duration_pMS[primary] = tmp
        t_MS_tmp = copy.copy(t_MS_m)
        if compute_binaries:
            tmp = t_MS_tmp[primary]
            tmp[present_state_m1 == 53] = t_MS_merger[present_state_m1 == 53]
            t_MS_tmp[primary] = tmp
        actual_scaled_times_pMS = (apparent_age_m[ind_pMS_present]-t_MS_tmp[ind_pMS_present])/duration_pMS[ind_pMS_present]


        if compute_binaries:
            # # # POST MAIN SEQUENCE: secondaries # # #
            # Pick the regular post main sequence secondary stars and accretors
            ind_pMS_present_m2 = stars_present_m2*(present_state_m2 == 3)

            # Calculate their scaled ages
            #duration_pMS_m2 = t_lifetime_m2-t_MS_m2
            #ind_acc = (present_state_m1 >= 2)*(present_state_m2 == 3)
            #duration_pMS_m2[ind_acc] = t_lifetime_acc[ind_acc] - t_MS_acc[ind_acc]
            #t_MS_tmp_m2 = copy.copy(t_MS_m2)
            #t_MS_tmp_m2[ind_acc] = t_MS_acc[ind_acc]
            #actual_scaled_times_pMS_m2 = ((apparent_age_m2[ind_pMS_present_m2]-t_MS_tmp_m2[ind_pMS_present_m2])/
            #                              duration_pMS_m2[ind_pMS_present_m2])
            duration_pMS_m2 = t_lifetime_acc[ind_pMS_present_m2] - t_MS_acc[ind_pMS_present_m2]    # I think this should also work for stars that have not accreted
            actual_scaled_times_pMS_m2 = (apparent_age_m2[ind_pMS_present_m2]-t_MS_acc[ind_pMS_present_m2])/duration_pMS_m2



            # # # STRIPPED STARS: primaries only # # #
            ind_strip_present = stars_present*(present_state == 2)

            # Calculate their scaled ages
            duration_strip = t_lifetime_m-t_MS_m
            actual_scaled_times_strip = ((apparent_age_m[ind_strip_present]-t_MS_m[ind_strip_present])/
                                         duration_strip[ind_strip_present])

        # Tell the log
        #fid_log = open(log_name,'a')
        #fid_log.write('Preparing for interpolation.... \n')
        #fid_log.close() 



        # Storage space for properties
        # Primaries and single stars
        # Main sequence
        MESA_params_MS_interp = np.zeros([nbr_MESA_param,np.sum(ind_MS_present)])

        # Post main sequence
        MESA_params_pMS_interp = np.zeros([nbr_MESA_param,np.sum(ind_pMS_present)])

        # --- UPDATE HERE ---
        if compute_binaries:
            # Secondaries
            # Main sequence
            MESA_params_MS_interp_m2 = np.zeros([nbr_MESA_param_b,np.sum(ind_MS_present_m2)])
            # Post main sequence
            MESA_params_pMS_interp_m2 = np.zeros([nbr_MESA_param_b,np.sum(ind_pMS_present_m2)])

            # Stripped stars
            MESA_params_strip_interp = np.zeros([nbr_MESA_param_b,np.sum(ind_strip_present)])



        # Find the closest scaled times in the array t_scaled
        # Primaries and single stars
        # Main sequence
        abs_tmp = np.abs(((actual_scaled_times_MS*np.ones([num_t,1])).T) - t_scaled*np.ones([np.sum(ind_MS_present),1]))
        ind_t_scaled_MS = ((np.min(abs_tmp,axis=1)*np.ones([num_t,1])).T) == abs_tmp
        # Post main sequence
        abs_tmp = np.abs(((actual_scaled_times_pMS*np.ones([num_t,1])).T) - t_scaled*np.ones([np.sum(ind_pMS_present),1]))
        ind_t_scaled_pMS = ((np.min(abs_tmp,axis=1)*np.ones([num_t,1])).T) == abs_tmp

        # Tell the log
        #fid_log = open(log_name,'a')
        #fid_log.write('Preparing for interpolation 2.... \n')
        #fid_log.close() 


        if compute_binaries:
            # Secondaries
            # Main sequence
            abs_tmp = np.abs(((actual_scaled_times_MS_m2*np.ones([num_t,1])).T) - t_scaled*np.ones([np.sum(ind_MS_present_m2),1]))
            ind_t_scaled_MS_m2 = ((np.min(abs_tmp,axis=1)*np.ones([num_t,1])).T) == abs_tmp
            # Post main sequence
            abs_tmp = np.abs(((actual_scaled_times_pMS_m2*np.ones([num_t,1])).T) - t_scaled*np.ones([np.sum(ind_pMS_present_m2),1]))
            ind_t_scaled_pMS_m2 = ((np.min(abs_tmp,axis=1)*np.ones([num_t,1])).T) == abs_tmp


            # Stripped stars
            abs_tmp = np.abs(((actual_scaled_times_strip*np.ones([num_t,1])).T) - t_scaled*np.ones([np.sum(ind_strip_present),1]))
            ind_t_scaled_strip = ((np.min(abs_tmp,axis=1)*np.ones([num_t,1])).T) == abs_tmp



        # Loop over the scaled times, they are fewer than the stars
        for i in range(num_t):

            # Find the stars that have been associated with this scaled time
            # We are going to loop over mass
            # Primaries and single stars
            # Main sequence 
            mtmp = m[ind_MS_present][ind_t_scaled_MS[:,i]]

            for cc in range(nbr_MESA_param):

                # 1D interpolation for main sequence stars if not magnetic
                if compute_magnetic == False:
                    MESA_params_MS_interp[cc,ind_t_scaled_MS[:,i]] = np.interp(mtmp,m_grid,MESA_params_MS_scaled[cc,:,i])
                    MESA_params_MS_interp[cc,ind_t_scaled_MS[:,i]] = extrapolate(mtmp,MESA_params_MS_interp[cc,ind_t_scaled_MS[:,i]],m_grid,MESA_params_MS_scaled[cc,:,i])

                # 2D interpolation for magnetic stars - need both M and B
                else:

                    Btmp = Binit[ind_MS_present][ind_t_scaled_MS[:,i]]

                    # Create the 2D interpolation function
                    z = MESA_params_MS_scaled[cc,:,:,i]
                    f = interpolate.RegularGridInterpolator((m_grid_B,B_strength_grids), z.T)

                    # Use the function to assign the interpolated value
                    ind_m = (mtmp>np.min(m_grid_B))*(mtmp<np.max(m_grid_B))
                    xy_new = list(zip(mtmp[ind_m],Btmp[ind_m]))

                    tmp = np.zeros(len(mtmp))
                    tmp[ind_m] = f(xy_new)

                    MESA_params_MS_interp[cc,ind_t_scaled_MS[:,i]] = tmp




            if exclude_pMS == False:
                # Post main sequence  -- UPDATE: there is no possibility for post-MS magnetic stars
                mtmp = m[ind_pMS_present][ind_t_scaled_pMS[:,i]]
                for cc in range(nbr_MESA_param):
                    MESA_params_pMS_interp[cc,ind_t_scaled_pMS[:,i]] = np.interp(mtmp,m_grid[reach_he_depletion_sin],MESA_params_pMS_scaled[cc,reach_he_depletion_sin,i])
                    MESA_params_pMS_interp[cc,ind_t_scaled_pMS[:,i]] = extrapolate(mtmp,MESA_params_pMS_interp[cc,ind_t_scaled_pMS[:,i]],m_grid[reach_he_depletion_sin],MESA_params_pMS_scaled[cc,reach_he_depletion_sin,i])



            # --- UPDATE HERE ---
            if compute_binaries:

                # Secondaries
                # These are allowed to be of lower mass than Mmin for stripping, so therefore I need to allow for extrapolation 
                # Main sequence
                mtmp = m2[ind_MS_present_m2][ind_t_scaled_MS_m2[:,i]]
                for cc in range(nbr_MESA_param):
                    MESA_params_MS_interp_m2[cc,ind_t_scaled_MS_m2[:,i]] = np.interp(mtmp,m_grid,MESA_params_MS_scaled[cc,:,i])
                    MESA_params_MS_interp_m2[cc,ind_t_scaled_MS_m2[:,i]] = extrapolate(mtmp,MESA_params_MS_interp_m2[cc,ind_t_scaled_MS_m2[:,i]],m_grid,MESA_params_MS_scaled[cc,:,i])


                if exclude_pMS == False:
                    # Post main sequence
                    mtmp = m2[ind_pMS_present_m2][ind_t_scaled_pMS_m2[:,i]]
                    for cc in range(nbr_MESA_param):
                        MESA_params_pMS_interp_m2[cc,ind_t_scaled_pMS_m2[:,i]] = np.interp(mtmp,m_grid[reach_he_depletion_sin],MESA_params_pMS_scaled[cc,reach_he_depletion_sin,i])
                        MESA_params_pMS_interp_m2[cc,ind_t_scaled_pMS_m2[:,i]] = extrapolate(mtmp,MESA_params_pMS_interp_m2[cc,ind_t_scaled_pMS_m2[:,i]],m_grid[reach_he_depletion_sin],MESA_params_pMS_scaled[cc,reach_he_depletion_sin,i])


                    # Stripped stars
                    # Interpolating over the initial masses
                    # I need to also allow for extrapolation here since some of the stripped star models don't reach central helium depletion
                    mtmp = m[ind_strip_present][ind_t_scaled_strip[:,i]]
                    if (len(mtmp) > 0):
                        for cc in range(nbr_MESA_param_b):
                            MESA_params_strip_interp[cc,ind_t_scaled_strip[:,i]] = np.interp(mtmp,m_bin_grid,MESA_params_strip_scaled[cc,:,i])
                            MESA_params_strip_interp[cc,ind_t_scaled_strip[:,i]] = extrapolate(mtmp,MESA_params_strip_interp[cc,ind_t_scaled_strip[:,i]],m_bin_grid,MESA_params_strip_scaled[cc,:,i])            


        # Tell the log
        fid_log = open(log_name,'a')
        fid_log.write('Used the interpolation scheme to find properties of the stars \n')
        if compute_binaries:
            fid_log.write('I extrapolated the properties for the secondaries with M < 2 Msun\n')
            fid_log.write('I also allowed for extrapolation to higher initial masses for stripped stars\n')
        fid_log.close() 




        # Actually stick these interpolated properties into the large array containing all types of stars
        # # #  MAIN SEQUENCE # # # 
        # Primaries and single stars
        for cc in range(nbr_MESA_param):
            MESA_params_interp[cc,ind_MS_present] = MESA_params_MS_interp[cc,:]

        # --- UPDATE HERE ---
        if compute_binaries:
            # Secondaries
            for cc in range(nbr_MESA_param):
                MESA_params_interp_m2[cc,ind_MS_present_m2] = MESA_params_MS_interp_m2[cc,:]

        # Tell the log
        fid_log = open(log_name,'a')
        fid_log.write('Main sequence star properties done for single stars and binaries. Evaluated at '+str(eval_time/1e6)+'Myr\n')
        fid_log.close() 

        # # # POST MAIN SEQUENCE # # # 
        # Primaries and single stars
        if exclude_pMS == False:
            for cc in range(nbr_MESA_param):
                MESA_params_interp[cc,ind_pMS_present] = MESA_params_pMS_interp[cc,:]

        # The secondaries, post-main sequence phase
        if compute_binaries:
            # Secondaries
            if exclude_pMS == False:
                for cc in range(nbr_MESA_param):
                    MESA_params_interp_m2[cc,ind_pMS_present_m2] = MESA_params_pMS_interp_m2[cc,:]

        # Tell the log
        fid_log = open(log_name,'a')
        fid_log.write('Post main sequence star properties done for single stars and binaries. Evaluated at '+str(eval_time/1e6)+'Myr\n')
        fid_log.close() 

        # The stripped star phase - always post main sequence
        if compute_binaries:
            # # # STRIPPED STARS # # # 
            if exclude_pMS == False:
                for cc in range(nbr_MESA_param):
                    cc2 = col_bin.index(col[cc])
                    MESA_params_interp[cc,ind_strip_present] = MESA_params_strip_interp[cc2,:]

            # Tell the log
            fid_log = open(log_name,'a')
            fid_log.write('Stripped star properties done!. Evaluated at '+str(eval_time/1e6)+'Myr\n')
            fid_log.close() 


        if iii == 0:
            if save_figs:
                if compute_magnetic == False:
                    # # # Verification figure
                    # Test the interpolation scheme by plotting the theoretical HR diagram
                    ww = 16
                    hh = 10
                    fig, (ax,ax2) = plt.subplots(1,2, figsize=(ww,hh))

                    xlim = [3.4,5.1]
                    ylim = [-0.2,6.5]

                    # Some parameters needed for the HRD
                    log_Teff_interp = MESA_params_interp[col.index('log_Teff'),:]
                    log_L_interp = MESA_params_interp[col.index('log_L'),:]

                    # # # Left panel
                    # Main sequence stars
                    ax.plot(log_Teff_interp[ind_MS_present],log_L_interp[ind_MS_present],'.k',label='MS Primaries, \n singles \n and mergers')
                    #  --- UPDATE HERE ---
                    if compute_binaries:
                        ax.plot(log_Teff_interp_m2[ind_MS_present_m2],log_L_interp_m2[ind_MS_present_m2],'.',color=[.4]*3,
                                label='MS Secondaries')

                    # post Main sequence stars
                    if exclude_pMS == False:
                        limegreen = [0,1,0]
                        forest = [0,.5,0]
                        ax.plot(log_Teff_interp[ind_pMS_present],log_L_interp[ind_pMS_present],'.',color=limegreen,
                                label='post-MS primaries, \n singles and \n mergers')
                        #  --- UPDATE HERE ---
                        if compute_binaries:
                            ax.plot(log_Teff_interp_m2[ind_pMS_present_m2],log_L_interp_m2[ind_pMS_present_m2],'.',color=forest, 
                                label='post-MS secondaries')

                        # Stripped stars
                        purple = [.5,0,.8]
                        #  --- UPDATE HERE ---
                        if compute_binaries:
                            ax.plot(log_Teff_interp[ind_strip_present],log_L_interp[ind_strip_present],'.',color=purple,label='Stripped stars')

                    # Single star evolutinary models
                    for i in range(nbr_sims):

                        log_Teff = MESA_params[i][col.index('log_Teff')]
                        log_L = MESA_params[i][col.index('log_L')]
                        center_he4 = MESA_params[i][col.index('center_he4')]

                        clr = [.7]*3
                        if m_grid[i] < 51.:
                            if reach_he_depletion_sin[i]:
                                clr='r'
                                ind_he = center_he4 > 0.01
                                ax.plot(log_Teff,log_L,'-',color=[.7]*3)
                                ax.plot(log_Teff[ind_he],log_L[ind_he],'-',color=clr)
                            else:
                                ax.plot(log_Teff,log_L,'-',color=clr)
                        if reach_he_depletion_sin[i]==False:
                            ax.plot(log_Teff[ind_ZAMS[i]:ind_TAMS[i]],log_L[ind_ZAMS[i]:ind_TAMS[i]],'-r')


                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    ax.legend(loc=3,fontsize=0.7*fsize,edgecolor='none')
                    ax.invert_xaxis()
                    ax.set_ylabel('$\\log_{10} (L/L_{\\odot})$')
                    ax.set_xlabel('$\\log_{10} (T_{\\mathrm{eff}}/\\mathrm{K})$')
                    ax.tick_params(direction="in", which='both')
                    ax.xaxis.set_ticks_position('both')
                    ax.yaxis.set_ticks_position('both')
                    ax.tick_params('both', length=6, width=1, which='major')


                    # # # Right panel
                    # Main sequence stars
                    ax2.plot(log_Teff_interp[ind_MS_present],log_L_interp[ind_MS_present],'.k')
                    #  --- UPDATE HERE ---
                    if compute_binaries:
                        ax2.plot(log_Teff_interp_m2[ind_MS_present_m2],log_L_interp_m2[ind_MS_present_m2],'.',color=[.4]*3)

                    if (exclude_pMS == False):
                        # post Main sequence stars
                        ax2.plot(log_Teff_interp[ind_pMS_present],log_L_interp[ind_pMS_present],'.',color=limegreen)
                        #  --- UPDATE HERE ---
                        if compute_binaries:
                            ax2.plot(log_Teff_interp_m2[ind_pMS_present_m2],log_L_interp_m2[ind_pMS_present_m2],'.',color=forest)

                        # Stripped stars
                        #  --- UPDATE HERE ---
                        if compute_binaries:
                            ax2.plot(log_Teff_interp[ind_strip_present],log_L_interp[ind_strip_present],'.',color=purple)

                    # Stripped star evolutionary models
                    #  --- UPDATE HERE ---
                    if compute_binaries:
                        for i in range(nbr_sims_bin):
                            clr='r'  #[.7]*3
                            #if reach_he_depletion_strip[i]:
                            #    clr='r'
                            indices = np.arange(len(star_age_b[i]))
                            if center_he4_b[i][-1] < 0.01:
                                ind_he = indices[center_he4_b[i] > 0.01][-1]
                            else:
                                ind_he = indices[-1]
                            ind_tmp = RLOF_b[i]*(center_he4_b[i] > 0.5)
                            if m_bin_grid[i] < 4.:
                                if center_he4_b[i][-1] < 0.01:
                                    ind_tmp = ((RLOF_b[i]+(log_abs_mdot_b[i] > -9))>0)*(center_he4_b[i] > 0.5)
                                else:
                                    ind_tmp = ((RLOF_b[i]+(log_abs_mdot_b[i] > -9))>0)*(center_he4_b[i] > 0.5)

                            ind_finish_RLOF = indices[ind_tmp][-1]
                            ind_strip_bin_tmp = indices[ind_finish_RLOF:ind_he]
                            #ind = center_he4_b[i] > 0.01
                            ax2.plot(log_Teff_b[i],log_L_b[i],'-',color=[.7]*3)
                            ax2.plot(log_Teff_b[i][ind_strip_bin_tmp],log_L_b[i][ind_strip_bin_tmp],'-',color=clr)

                    ax2.set_xlim(xlim)
                    ax2.set_ylim(ylim)
                    #ax2.legend(loc=3,fontsize=0.7*fsize,edgecolor='none')
                    ax2.invert_xaxis()
                    ax2.set_ylabel('$\\log_{10} (L/L_{\\odot})$')
                    ax2.set_xlabel('$\\log_{10} (T_{\\mathrm{eff}}/\\mathrm{K})$')
                    ax2.tick_params(direction="in", which='both')
                    ax2.xaxis.set_ticks_position('both')
                    ax2.yaxis.set_ticks_position('both')
                    ax2.tick_params('both', length=6, width=1, which='major')

                    fig.savefig(plots_dir+'/MESA_interpolation_popHRD.png',format='png',bbox_inches='tight',pad_inches=0.1)
                    plt.close(fig)


                    # Tell the log
                    fid_log = open(log_name,'a')
                    fid_log.write('Saved an HRD for the population\n\n')
                    fid_log.close() 



        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        #                                                                             #
        #     Small post-processing part                                              #
        #       - give masses and radii to WDs (I don't think there are NSs or BHs)   #
        #       - classify objects that overfill Roche-lobe this moment differently   #
        #                                                                             #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  

        # The compact objects
        # primaries and singles
        ind_co = ((present_state == 4)+(present_state == 54)) > 0
        star_mass_interp = MESA_params_interp[col.index('star_mass'),:]
        star_mass_interp[ind_co*(m < 9.)] = 0.6     # white dwarfs
        star_mass_interp[ind_co*(m >= 9.)*(m < 15.)] = 1.4   # neutron stars
        star_mass_interp[ind_co*(m > 15.)] = 5.   # black holes
        MESA_params_interp[col.index('star_mass'),:] = copy.copy(star_mass_interp)


        #  --- UPDATE HERE ---
        if compute_binaries:
            # secondaries 
            ind_co_m2 = present_state_m2 == 4
            star_mass_interp_m2 = MESA_params_interp_m2[col.index('star_mass'),:]
            star_mass_interp_m2[ind_co_m2*(m2 < 9.)] = 0.6    # white dwarfs  
            star_mass_interp_m2[ind_co_m2*(m2 >= 9.)*(m2 < 15.)] = 1.4    # neutron stars
            star_mass_interp_m2[ind_co_m2*(m2 > 15.)] = 5.     # black holes

            m1current = star_mass_interp[primary]
            m2current = copy.copy(star_mass_interp_m2)


            # I will also assume that the white dwarfs have a radius of 0.005 RSun
            log_R_interp_m2 = MESA_params_interp_m2[col.index('log_R'),:]
            log_R_interp_m2[ind_co_m2*(m2 < 9.)] = np.log10(0.005)

            # Get all the systems that are currently having a stripped star
            ind_sn = (present_state_m1 == 2)*ind_strip

            # I want to also flag systems that currently are too big for their periods
            # These are: (1) contraction of stripped stars not accounted for (artificial), (2) MS secondaries that were larger than ZAMS at interaction and therefore don't fit (should have merged), (3) MS and pMS secondaries that filled their Roche-lobes after the first interaction was finished (physical, but no idea what should happen to them). 
            if np.sum(ind_sn):
                q1 = m1current[ind_sn]/m2current[ind_sn]
                q2 = m2current[ind_sn]/m1current[ind_sn]
                rL1 = 0.49*(q1**(2./3.))/(0.69*(q1**(2./3.)) + np.log(1.+(q1**(1./3.))))  # Unitless
                rL2 = 0.49*(q2**(2./3.))/(0.69*(q2**(2./3.)) + np.log(1.+(q2**(1./3.))))  # Unitless
                atmp = ((Pfinal[ind_sn]**2.)*(G*(m1current[ind_sn]+m2current[ind_sn])/(4.*(np.pi**2.))))**(1./3.)   # AU
                atmp = atmp*u.AU.to(u.R_sun)   # RSun
                RL1 = rL1*atmp    # Roche-lobe size of stripped star in RSun
                RL2 = rL2*atmp    # Roche-lobe size of companion star in RSun

                # Get the predicted radii of the stars
                log_R_interp = MESA_params_interp[col.index('log_R'),:]
                R1 = 10**log_R_interp[primary]   # RSun
                R2 = 10**log_R_interp_m2

                # (1) stripped stars that are so young that they are interpolated to be contracting, but should be smaller given their evolutionary history
                ind_overfill_strip = R1[ind_sn] > RL1

                # (2) and (3) 
                ind_overfill_m2 = R2[ind_sn] > RL2
                ind_overfill_MS_m2 = ind_overfill_m2*(present_state_m2[ind_sn] == 1)
                ind_overfill_pMS_m2 = ind_overfill_m2*(present_state_m2[ind_sn] == 3)

                # Now, reflag these systems
                # For the post-MS secondaries that fill the Roche lobe when having a stripped companion, I will give them star_state = -3 for un-known post-MS
                tmp = present_state_m2[ind_sn]
                tmp[ind_overfill_pMS_m2] = -3
                # For the MS secondaries that fill the Roche lobe, I flag as -1
                tmp[ind_overfill_MS_m2] = -1
                present_state_m2[ind_sn] = copy.copy(tmp)
                # For the stripped stars that fill the Roche lobe, I flag as -2
                tmp = present_state_m1[ind_sn]
                tmp[ind_overfill_strip] = -2
                present_state_m1[ind_sn] = copy.copy(tmp)
                present_state[primary] = copy.copy(present_state_m1)

                # Tell the log that the star-states are updated because of overfilling
                fid_log = open(log_name,'a')
                fid_log.write('Updated the star-states of stars in systems that fill their Roche lobe (adding a minus).\n\n')
                fid_log.write('There are '+str(np.sum(present_state_m2 < 0))+' cases of M2 overfilling...\n')
                fid_log.close() 


        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        #                                                                             #
        #     Write a data-file with stellar properties                               #
        #                                                                             #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
        # Write a file with the properties of the stars that are present in the stellar population
        # This is the first data-file that is produced by the code. It will contain properties predicted by MESA.

        # Initiate the file
        filename_run = 'data1_'+run_name+'_'+str(eval_time)+'yrs.txt'
        if iii == 0:
            fid = open(filename_run,'w')
        else:
            fid = open(filename_run,'a')

        if iii == 0:
            # Write a header
            fid.write('#\n')
            fid.write('#     Data from population synthesis for '+run_name+'\n')
            fid.write('#\n')

            date = str(datetime.datetime.now()).split(' ')[0]
            fid.write('# By Ylva Gotberg, '+date+'\n')
            fid.write('#\n')

            fid.write('# Initial conditions:\n')
            fid.write('#   Using the single star models from '+loc_sin_grid+' to determine the radii of donors at different evolutionary stages\n')
            fid.write('#   This means that we assume metallicity, Z='+str(Z)+'\n')
            fid.write('#   Using the lifetimes of stars from single star models in'+loc_sin_grid+'\n')
            fid.write('#   IMF: '+IMF_choice+', M_min = '+str(mmin)+' Msun, M_max = '+str(mmax)+' Msun \n')
            fid.write('# We assume the star-formation type: '+type_SF+',')
            if type_SF == 'constant':
                fid.write(' with a star-formation rate of '+str(starformation_rate)+' Msun/yr\n')
            elif type_SF == 'starburst':
                fid.write(' with a total mass of stars of '+str(total_mass_starburst)+' Msun\n')
            fid.write('# This file contains the information about the stars that are present in the population after '+str(eval_time)+' years\n')
            if compute_binaries:
                fid.write('#   fbin:'+fbin_choice+'\n')
                fid.write('#   q distribution: '+q_choice+', '+str(qmin)+' < q < '+str(qmax)+'\n')
                if P_choice == 'Opik_Sana':
                    fid.write('#   P distribution: Sana+12 for M1>15 Msun and Opik24 for M1<15 Msun\n')
                fid.write('#\n')

                fid.write('# Physical assumptions:\n')
                fid.write('#   Critical mass ratio for the donor main sequence, q_crit_MS='+str(q_crit_MS)+'\n')
                fid.write('#   Critical mass ratio for the donor Hertzsprung gap, q_crit_HG='+str(q_crit_HG)+'\n')
                if alpha_prescription:
                    fid.write('#   Using the alpha-prescription for common envelope evolution\n')
                    fid.write('#   alpha_CE = '+str(alpha_CE)+', lambda_CE = '+str(lambda_CE)+'\n')
                fid.write('#   Using the binary star models from '+loc_bin_grid+' to determine the masses and radii of stripped stars\n')
                fid.write('#   Stripped stars can only be formed from donor stars with '+str(Minit_strip_min)+' < M1init/Msun < '+str(Minit_strip_max)+'\n')
                fid.write('#   Actually, we assume that binary interaction only can occur in the above mass range.\n')
                fid.write('#   For stripped stars we set the total lifetime to tau_MS+tau_strip, where the duration of the stripped phases (tau_strip) come from the binary models in'+loc_bin_grid+'\n')
                fid.write('#   Mass transfer efficiency is set to beta_MS = '+str(beta_MS)+', beta_HG = '+str(beta_HG)+', beta_HG_CEE = '+str(beta_HG_CEE)+'\n')
                fid.write('#   For mergers, we assume that their final mass is M1init+M2init\n')
                fid.write('#   The final period is set by the treatment of angular momentum during interaction, we ignore the effect of wind mass loss on the period.\n')
                fid.write('#   We use the '+angmom+' setting for the treatment of angular momentum.\n')
                if 'gamma' in angmom:
                    fid.write('#   For RLOF, we assume that gamma_MS = '+str(gamma_MS)+' and gamma_HG'+str(gamma_HG)+'\n')
                if alpha_prescription:
                    fid.write('#   For common envelope, the period is set in the alpha prescription\n')
                fid.write('#   Rejuvenation of mass gainers and MSMS mergers is set by '+rejuv_choice+'\n')
                fid.write('#   HGMS mergers are not rejuvenated, but their lifetimes are adapted to the new mass\n')
                fid.write('#\n')
                fid.write('# Even though the full population synthesis accounts for the full mass range from '+str(mmin)+' to '+str(mmax)+' Msun, \n# I will only keep the stars with initial masses (Minit or M1init) between '+str(Minit_strip_min)+' and '+str(Minit_strip_max)+' Msun.\n')
            fid.write('# I am ignoring single stars that died and binaries in which at least the primary died.\n')
            fid.write('# I show dead secondaries as long as the primary is still alive\n')
            fid.write('# For the star state: 1 - main-sequence star, 2 - stripped star, 3 - post main-sequence star, 4 - compact object, 51 - merger product in MS stage.\n')

            # If this switch is on, it only records stripped star systems (stripped now)
            if record_stars == 'stripped_star_systems_only':
                fid.write('# In this output, we only output the stripped star systems.\n')

            # After that massive header, I will put a second header for the different quantities
            fid.write('#\n')
            if compute_binaries:
                header_data1 = '# Star_ID \t Star_state_m1 \t Star_state_m2 \t Evolution \t M1init \t M2init \t Pinit \t Pcurrent'
                for cc in range(nbr_MESA_param):
                    header_data1 = header_data1+' \t '+col_bin[cc]+'_1 \t '+col[cc]+'_2'
            elif compute_magnetic: 
                header_data1 = '# Star_ID \t Star_state \t Minit \t Binit'
                for cc in range(nbr_MESA_param):
                    header_data1 = header_data1+' \t '+col[cc]
            else:
                header_data1 = '# Star_ID \t Star_state \t Minit'
                for cc in range(nbr_MESA_param):
                    header_data1 = header_data1+' \t '+col[cc]

            fid.write(header_data1+'\n')


        # Now, I will print the actual data
        indices_b = np.arange(np.sum(primary))

        inds = np.arange(len(primary))[stars_present]

        # Loop over the stars
        for k in range(np.sum(stars_present)):

            write_str = ''

            i = inds[k]

            # Is it a single star?, also is it not dead and within the mass limits
            if single[i] and (present_state[i] != 4) and (m[i]> minimum_mass_to_print) and (m[i] < maximum_mass_to_print):

                if compute_binaries:
                    write_str = str(i)+'\t'+str(int(present_state[i]))+'\t - \t single \t'+'%10.5f'% m[i]+'\t - \t - \t -'
                    for cc in range(nbr_MESA_param):
                        write_str = write_str+' \t '+'%10.5f' % (MESA_params_interp[cc,i]) + ' \t -'
                elif compute_magnetic:
                    write_str = str(i)+'\t'+str(int(present_state[i])) + '\t'+'%10.5f'% m[i] + '\t'+'%10.5f'% Binit[i]
                    for cc in range(nbr_MESA_param):
                        if col[cc] == 'star_mdot':
                            write_str = write_str+' \t '+'%10.5e' % (MESA_params_interp[cc,i])
                        else:
                            write_str = write_str+' \t '+'%10.5f' % (MESA_params_interp[cc,i])  
                else: 
                    write_str = str(i)+'\t'+str(int(present_state[i])) + '\t'+'%10.5f'% m[i]
                    for cc in range(nbr_MESA_param):
                        write_str = write_str+' \t '+'%10.5f' % (MESA_params_interp[cc,i])                
                write_str = write_str+'\n'


            # Is it a binary star?
            # --- UPDATE HERE ---
            elif primary[i] and (present_state[i] != 4):
                # The index in the binary system arrays
                j = np.sum(primary[:i+1])-1

                # Predicted evolution of the binary
                evolution = 'not_interacting'
                if ind_merge[j]:
                    evolution = 'merger'
                    if ind_mergeA[j]:
                        evolution = evolution+'_MS'
                    elif ind_mergeB[j]:
                        evolution = evolution+'_HG'
                    elif ind_mergeC[j]:
                        evolution = evolution+'_postHG'
                if ind_caseA[j]:
                    evolution = 'strip_RLOF_MS'
                elif ind_caseB[j]:
                    evolution = 'strip_RLOF_HG'
                elif ind_caseB_CEE[j]:
                    evolution = 'strip_CEE_HG'
                elif ind_caseC[j]:
                    evolution = 'strip_postHG'

                # Decide which period to print
                # If the star can have interacted
                if (present_state_m1[j] > 1):
                    current_P = Pfinal[j]
                # Catch the mergers
                elif (present_state_m1[j] == 1) and (present_state_m2[j] != 1):
                    current_P = '-'
                # The ones that haven't interacted
                else:
                    current_P = P[j]

                # The age of the star at the moment of writing the file
                #actual_age = eval_time-birthday_m1[j]

                # Prepare the string to write to file
                write_str = str(i)+'\t'+str(int(present_state[i]))+'\t'+str(int(present_state_m2[j]))+'\t'+evolution+'\t %10.5f' % m1[j]+'\t %10.5f' % m2[j]+'\t %10.5f' % P[j]+'\t %10.5f'% current_P
                for cc in range(nbr_MESA_param):
                    write_str = write_str+' \t '+'%10.5f' % (MESA_params_interp[cc,i]) + ' \t '+'%10.5f' % (MESA_params_interp_m2[cc,j])
                write_str = write_str+'\n'

                # In case the output should only contain the stripped star systems 
                if record_stars == 'stripped_star_systems_only':
                    if present_state_m1[j] == 2.:
                        fid.write(write_str)

            # In case all stars in the system should be output.
            if record_stars == 'all':
                fid.write(write_str)


        fid.close()


        # Tell the log
        fid_log = open(log_name,'a')
        fid_log.write('Wrote to data file #1 ('+filename_run+')... \n')
        fid_log.write('This file contains properties predicted by MESA.\n\n')
        fid_log.close() 


    
"""
# # # # # THE ZAMS # # # # # 
# This does not need to be in the main loop
# # The location of the ZAMS in terms of stellar properties
filename_ZAMS = 'ZAMS_properties_Z'+str(Z)+'_evol.txt'
fid = open(filename_ZAMS,'w')
fid.write('# This file contains the stellar properties of the zero-age main sequence.\n')
fid.write('# It is for MESA single star models with Z='+str(Z)+'.\n')
#fid.write('# Minit \t log_Teff \t log_R \t log_L \t log_g\n')
header_ZAMS = '# Minit'
for cc in range(nbr_MESA_param):
    header_ZAMS=header_ZAMS+' \t '+col[cc]
fid.write(header_ZAMS+'\n')
for i in range(nbr_sims):
    #write_str = str(m_grid[i])+'\t'+str(log_Teff[i][ind_ZAMS[i]])+'\t'+str(log_R[i][ind_ZAMS[i]])+'\t'+str(log_L[i][ind_ZAMS[i]])+'\t'+str(log_g[i][ind_ZAMS[i]])+'\n'
    write_str = str(m_grid[i])
    for cc in range(nbr_MESA_param):
        write_str = write_str+' \t '+'%10.5e' % (MESA_params[i][cc][ind_ZAMS[i]])
    fid.write(write_str+'\n')
fid.close()

# Tell the log
fid_log = open(log_name,'a')
fid_log.write('Wrote also a file for the ZAMS properties predicted by MESA ('+filename_ZAMS+') \n')
fid_log.close() 
"""

# Tell the log
fid_log = open(log_name,'a')
fid_log.write('PART 1 OF MOSS FINISHED! \n')
fid_log.close() 

