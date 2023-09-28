#!/usr/bin/env python
# coding: utf-8

"""
#                                                                             #
#                        MOSS, part 2                                         # 
#                                                                             #
#                                                                             #
# version 1.0                                                                 #
# Ylva Gotberg, 8 October 2019                                                #
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
# This script contains the spectral synthesis part of MOSS. I interpolate the #
# spectral models to match each star. The model grids used are Gotberg+18 and #
# the Kurucz ATLAS9 models. This leaves a part of parameter space un-probed - #
# the contracting (and expanding, although not yet included in MOSS1)         #
# stripped stars.                                                             #
#                                                                             #
# The output from this code is a data file with identification numbers and    #
# magnitudes for specific filters, calculated using both blackbody and        #
# spectral models.                                                            #
#                                                                             #
# - restructuring to work for very large data1-files - Ylva, 14 August 2020   #
# - giving up on saving spectra and just interpolate over magnitudes - Ylva, 17 August 2020
#                                                                             #
"""


# Packages needed for running
import numpy as np
import copy
import os
import datetime
import astropy.units as u
from astropy import constants as const
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
fsize = 20
rc('font',**{'family':'serif','serif':['Times'],'size'   : fsize})
rc('text', usetex=True)

from GetSpectra import GetSpectra
from GetCorners import GetCorners


# Get some constants etc
G = const.G.to('AU3/(M_sun d2)').value   # Gravitational constant in AU^3 / (Msun * day^2)
RSun_AU = (u.Rsun).to(u.AU)        # Solar radius in AU
RSun_SI = u.R_sun.to(u.m)          # Solar radius in meters
h_SI = const.h.value    # Planck's constant
c_SI = const.c.value    # Speed of light
kB_SI = const.k_B.value   # Stefan-Boltzmann's constant
pc_SI = u.pc.to(u.m)    # 1 pc expressed in meters
erg_SI = u.erg.to(u.J)   # 1 erg in Joules
# The distance of 10 pc
dist_10pc = 10*pc_SI



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                                                             #
#     Input                                                                   #
#                                                                             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# The below should be somewhere in an input file - update this!
# _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

# The name of the run (should be the same as for MOSS, part 1)
#run_name = 'run_Z002_test'    # THIS SHOULD BE INPUT
run_name = [name for name in os.listdir('./') if 'data1' in name][0].split('data1_')[1].split('.txt')[0]

# Where the ZAMS data is? 
#filename_ZAMS = 'ZAMS_properties_Z0.002_evol.txt'    # Input somehow? 
filename_ZAMS = [name for name in os.listdir('./') if 'ZAMS_properties' in name][0]

# Metallicity 
#Z = 0.002
Z = np.float_(filename_ZAMS.split('Z')[2].split('_')[0])

# THIS SHOULD BE INPUT
loc_Kurucz = '/data002/ygoetberg/spectral_libraries/Kurucz/gridm10odfnew/coubesm10k2odfnew.dat'

#  The location of the spectral models
if Z == 0.014:
    Kurucz_spec_loc = '/data002/ygoetberg/spectral_libraries/Kurucz/gridp00odfnew/fp00k2odfnew.pck'
    loc_strip = '/data002/ygoetberg/CMFGEN/models/updated_atm_paper/grid_014_cmf/'
elif Z == 0.006:
    Kurucz_spec_loc = '/data002/ygoetberg/spectral_libraries/Kurucz/gridm05odfnew/fm05k2odfnew.pck'
    loc_strip = '/data002/ygoetberg/CMFGEN/models/updated_atm_paper/grid_006_cmf/'
elif Z == 0.002:
    Kurucz_spec_loc = '/data002/ygoetberg/spectral_libraries/Kurucz/gridm10odfnew/fm10k2odfnew.pck'
    loc_strip = '/data002/ygoetberg/CMFGEN/models/updated_atm_paper/grid_002_cmf/'


plots_dir = 'verification_plots'
save_figs = False

# Give which filters to include - NOT SURE HOW TO DO THIS BEST: put the filter functions in the folder itself? centralize MOSS?
loc_filters = 'filters/'
filters = [name for name in os.listdir(loc_filters) if ('dat' in name)]
nbr_filters = len(filters)
filters = np.array(filters)


# Interpolation scheme for the stripped star models 
interp_scheme = 'loglin'  #'loglog'  #'linlog'   #'linlin'  # 

# Switch for saving only blackbody estimates
only_BB = False


# _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 


# Use also the same log-file
log_name = 'log_'+run_name+'.log'

# Initiate the log-file
fid_log = open(log_name,'a')
fid_log.write('\n\n       This is the log for '+run_name+', part 2\n\n')
fid_log.write('Now the date and time is '+str(datetime.datetime.now())+'\n\n')
fid_log.close()



# # # # GET THE FILTERS # # # #
# Get the transmission curves from the filters
lambda_filters = [None]*nbr_filters
trans_filters = [None]*nbr_filters
name_filters = [None]*nbr_filters
for i in range(nbr_filters):
    if 'superbit' in filters[i]:
        data = np.loadtxt(loc_filters+filters[i],delimiter=',')
        lambda_filters[i] = data[:,0]*10.   # Angstrom
        trans_filters[i] = data[:,1]
    else:
        data = np.loadtxt(loc_filters+filters[i])
        lambda_filters[i] = data[:,0]   # Angstrom
        trans_filters[i] = data[:,1]    # transmission curves
        

if save_figs:
    # Verification plot showing the filter functions 
    fig, ax = plt.subplots(1,1,figsize=(10,6))
    for i in range(len(filters)):
        ax.plot(lambda_filters[i],trans_filters[i],'-')
    ax.set_xlabel('Wavelength [AA]')
    ax.set_ylabel('Transmission')
    fig.savefig(plots_dir+'/filter_transmission.png',format='png',bbox_inches='tight',pad_inches=0.1)
    plt.close(fig)


# The zeropoints will all be 3631 Jy since we are using the AB magnitude system
zeropoint_Jy = 3631.
zeropoint = zeropoint_Jy*u.Jy.to('erg s^-1 cm^-2 Hz^-1')   # erg s^-1 cm^-2 Hz^-1  (same unit as F_nu)

# Frequency arrays for the filters in Hz
nu_filters = c_SI/(np.array(lambda_filters)*1e-10)   

    
# Tell the log that the filters were read
fid_log = open(log_name,'a')
write_str = 'Read the filter functions for'
for l in range(len(filters)):
    write_str = write_str+' '+filters[l]
fid_log.write(write_str+'\n')
fid_log.write('Using AB magnitude system\n')
fid_log.close()



# This is only read if the atmosphere models should be used
if  only_BB == False:
    # # # #   GET THE ATMOSPHERE GRID PARAMETERS   # # # #
    # Currently we are using Kurucz models for companions and atmosphere models for stripped stars
    # KURUCZ models
    data = np.loadtxt(loc_Kurucz,skiprows=2)
    T_Kurucz = data[:,1]
    logg_Kurucz = data[:,2]
    
    # STRIPPED STAR grids
    specs = [name for name in os.listdir(loc_strip) if ((name[0] == 'M') and ('M1_' in name))]
    nbr_stripped_models = len(specs)
    
    # Storage space for the properties of the stars in the stripped star spectral models
    Tstar_stripped = np.zeros(nbr_stripped_models)
    Lstar_stripped = np.zeros(nbr_stripped_models)
    logg_stripped = np.zeros(nbr_stripped_models)
    mass_stripped = np.zeros(nbr_stripped_models)
    
    # Get the temperature and log g
    for i in range(nbr_stripped_models):
        
        # Read from the MOD_SUM file
        fid = open(loc_strip+specs[i]+'/MOD_SUM','r')
        lines = fid.readlines()
        fid.close()
        for j in range(len(lines)):
            if 'T*' in lines[j]:
                Tstar_stripped[i] = np.float_(lines[j].split()[2].split('=')[1])
                logg_stripped[i] = np.float_(lines[j].split()[5].split('=')[1])
            if 'L*' in lines[j]:
                Lstar_stripped[i] = np.float_(lines[j].split()[0].split('=')[1])
                
        # Read the VADAT file to get the stripped star mass
        fid = open(loc_strip+specs[i]+'/VADAT')
        lines = fid.readlines()
        fid.close()
        for j in range(len(lines)):
            if '[MASS]' in lines[j]:
                mass_stripped[i] = np.float_(lines[j].split()[0])

    # Sort the arrays in order of mass
    ind_sort = np.argsort(mass_stripped)
    mass_stripped = mass_stripped[ind_sort]
    Tstar_stripped = Tstar_stripped[ind_sort]
    logg_stripped = logg_stripped[ind_sort]
    Lstar_stripped = Lstar_stripped[ind_sort]
    specs = np.array(specs)[ind_sort]

    # Tell the log that the atmosphere models were read
    fid_log = open(log_name,'a')
    fid_log.write('Atmospheres: read parameters for the Kurucz model grid and the stripped star model grid\n')
    fid_log.close()
   


    # # # # READ ATMOSPHERE MODELS # # # # 
    # Read the Kurucz spectra
    fid = open(Kurucz_spec_loc,'r')
    lines = fid.readlines()
    fid.close()

    # Get the wavelength array (it is given in nm)
    for i in range(len(lines)):
        if 'END' in lines[i]:
            ind1 = i+1
        if 'TEFF' in lines[i]:
            ind2 = copy.copy(i)
            break;
        
    tmp = lines[ind1:ind2]
    lambda_Kurucz = np.array([])
    for i in range(len(tmp)):
        lambda_Kurucz = np.concatenate([lambda_Kurucz,np.float_(tmp[i].split())])

    lambda_Kurucz = lambda_Kurucz*10.   # Putting the wavelengths in Ångström
    # Translate the wavelengths into frequency
    nu_Kurucz = c_SI/(lambda_Kurucz*1e-10)

    
    # Get all the models
    # Here are the indices where each model starts
    ind1s = []
    Teff_Kurucz = []
    log_g_Kurucz = []
    for i in range(len(lines)):
    
        if 'TEFF' in lines[i]:
            ind1s.append(i+1)
            Teff_Kurucz.append(np.float_(lines[i].split()[1]))
            log_g_Kurucz.append(np.float_(lines[i].split()[3]))
    Teff_Kurucz = np.array(Teff_Kurucz)
    log_g_Kurucz = np.array(log_g_Kurucz)
        
    # Get the spectra (intensities)
    # Unfortunately, the Kurucz models are given in Eddington flux (erg/cm^2/s/Hz/ster)
    Hnu_Kurucz = [None]*len(ind1s)
    Hnucont_Kurucz = [None]*len(ind1s)
    ds = 10.
    for i in range(len(ind1s)):
        specs_tmp = []
        if i < (len(ind1s)-1):
            i_end = int(ind1s[i+1]-1)
        else:
            i_end = int(len(lines))
        
        for k in range(ind1s[i],i_end):
            for j in range(int(len(lines[k])/ds)):
                specs_tmp.append(np.float_(lines[k][int(j*ds):int(j*ds+ds)]))
            
        Hnu_Kurucz[i] = specs_tmp[0:len(lambda_Kurucz)]
        Hnucont_Kurucz[i] = specs_tmp[len(lambda_Kurucz):2*len(lambda_Kurucz)]

    # Now, the Eddington flux is related to the intensity - makes sense
    Hnu_Kurucz_SI = np.array(Hnu_Kurucz)*erg_SI/(0.01**2.)
    Hnucont_Kurucz_SI = np.array(Hnucont_Kurucz)*erg_SI/(0.01**2.)
    Ilambda_Kurucz_SI = 4*c_SI*Hnu_Kurucz_SI/((lambda_Kurucz*1e-10)**2.) 
    Ilambdacont_Kurucz_SI = 4*c_SI*Hnucont_Kurucz_SI/((lambda_Kurucz*1e-10)**2.)
    # Convert to cgs
    Ilambda_Kurucz_cgs = Ilambda_Kurucz_SI*(u.W/(u.m**3.)).to('erg s-1 cm-2 AA-1')  
    Inu_Kurucz_cgs = lambda_Kurucz*Ilambda_Kurucz_cgs/nu_Kurucz
    nbr_Kurucz = len(Ilambda_Kurucz_SI)
    
    # Tell the log
    fid_log = open(log_name,'a')
    fid_log.write('Atmospheres: read in the intensities of the Kurucz spectral models.\n')
    fid_log.close()


    # Calculate the parameter that I can just scale with radius to get the absolute magnitudes later
    # mag = -2.5log10((pi/1e7)*(R/d)^2  * int(Inu*Trans dnu)/int(Trans dnu) / zeropoint)
    integral_Inu = np.zeros([nbr_Kurucz, nbr_filters])
    for l in range(nbr_filters):
        Ttmp = np.interp(lambda_Kurucz, lambda_filters[l], trans_filters[l])
        integral_Inu[:,l] = np.trapz(Inu_Kurucz_cgs*Ttmp, nu_Kurucz,axis=1)/np.trapz(trans_filters[l], nu_filters[l])


    # the maximum number of stars that the interpolation scheme can manage
    #max_num = 1000.

    # Tell the log
    #fid_log = open(log_name,'a')
    #fid_log.write('Atmospheres: Initialized a function for interpolating spectral models in the Kurucz grid.\n')
    #fid_log.write('-- There are some warnings for this function, I should check carefully at some point...\n')
    #fid_log.write('... setting the maximum number of stars that the interpolation scheme can manage to '+str(max_num)+'...\n')
    #fid_log.close()
 
    # Read the spectra of the stripped star models

    # Storage space for the actual spectra - might not be needed yet
    lambda_stripped_models = [None]*nbr_stripped_models
    Flambda_stripped_models = [None]*nbr_stripped_models
    Fcont_stripped_models = [None]*nbr_stripped_models

    # Storage for the stripped star model grid absolute magnitudes so I don't have to interpolate the spectra
    abs_mag_stripped_grid = np.zeros([nbr_stripped_models, nbr_filters])

    for i in range(nbr_stripped_models):
        # Get the spectra 
        lambda_stripped_models[i], Flambda_stripped_models[i], Fnorm, Fcont_stripped_models[i] = \
        GetSpectra(loc_strip+specs[i],'obs')
        nu_tmp = c_SI/(lambda_stripped_models[i]*1e-10)      # Calculate the frequency array
        Flambda_tmp = Flambda_stripped_models[i]*((1e3/10)**2.)    # Move the spectra to 10 pc distance
        Fnu_tmp = lambda_stripped_models[i]*Flambda_tmp/nu_tmp    # Convert to Fnu - good for magnitude calculation???
    
        # Calculate the absolute magnitudes
        for l in range(nbr_filters):
            Ttmp = np.interp(lambda_stripped_models[i], lambda_filters[l], trans_filters[l])
            Ftmp = np.trapz(Fnu_tmp*Ttmp,nu_tmp)/np.trapz(trans_filters[l], nu_filters[l])
            abs_mag_stripped_grid[i,l] = -2.5*np.log10(Ftmp/zeropoint)     
            

    # Tell the log
    fid_log = open(log_name,'a')
    fid_log.write('Atmospheres: Read the spectral energy distributions for stripped stars in the grid.\n')
    fid_log.write('Atmospheres: Calculated the absolute magnitudes for the stripped star model grid.\n')
    fid_log.close()
   
    # I am commenting the below - it was created for interpolating spectra
    # The interpolation for these stripped stars should just be done over the stripped star mass
    # I will need to put all stripped star spectra on the same wavelength ranges
    # I will put them on the same wavelength array as the Kurucz models
    #Flambda_stripped_models_coarse = np.zeros([nbr_stripped_models,len(lambda_Kurucz)])
    #for i in range(nbr_stripped_models):
    #    Flambda_stripped_models_coarse[i] = np.interp(lambda_Kurucz,lambda_stripped_models[i],Flambda_stripped_models[i])

    # Tell the log
    #fid_log = open(log_name,'a')
    #fid_log.write('Atmospheres: Made the SEDs for stripped stars coarser.\n')
    #fid_log.close()






# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                                                             #
#     ZAMS                                                                    #
#                                                                             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# The zero-age main sequence stars are in a small grid and this does therefore 
# not need a loop. I put it first for this reason. - Ylva 14 August 2020

            
# # # #  ZAMS: CREATE BLACKBODIES   # # # #

# Initiate a wavelength array (only need UV and optical, maybe some IR - double check)
lambda_AA = np.logspace(2.,4.7,1000)
lambda_SI = lambda_AA*1e-10
nu = c_SI/(lambda_AA*1e-10)

# Get the zero-age main-sequence
data = np.loadtxt(filename_ZAMS)
minit_ZAMS = data[:,0]
log_T_ZAMS = data[:,1]
log_R_ZAMS = data[:,2]
log_L_ZAMS = data[:,3]
log_g_ZAMS = data[:,4]
nbr_Z = len(log_T_ZAMS)

# Produce the intensity and flux
[llZ,TTZ] = np.meshgrid(lambda_SI,10**log_T_ZAMS)
[llZ,RRZ] = np.meshgrid(lambda_SI,(10**log_R_ZAMS)*u.R_sun.to(u.m))
B_lambda_Z = (2.*h_SI*(c_SI**2.)/(llZ**5.))/(np.exp(h_SI*c_SI/(llZ*kB_SI*TTZ))-1.)

F_lambda_Z_SI = np.pi*B_lambda_Z*((RRZ/dist_10pc)**2.)
F_lambda_Z_cgs = F_lambda_Z_SI/1e7

# Tell the log that blackbodies were created
fid_log = open(log_name,'a')
fid_log.write('ZAMS: Created blackbodies for the ZAMS.\n')
fid_log.close()



# # # #   ZAMS: CALCULATION OF THE ABSOLUTE MAGNITUDES - BLACKBODIES   # # # #

# Need to change the unit of the fluxes
F_nu_Z_cgs = lambda_AA*F_lambda_Z_cgs/nu

# Storage space for the blackbody absolute magnitudes
magnitudes_Z_BB = np.zeros([nbr_Z,nbr_filters])

# Loop over the filters and calculate the magntiudes for the stars for each filter
for l in range(nbr_filters):
    
    # Calculate the magnitude in the current band for all stars
    # This is the int(Fnu*T(nu))/int(T(nu)) part
    # More accurate with the wavelength array of the spectra below
    Ttmp = np.interp(lambda_AA, lambda_filters[l], trans_filters[l])
    FtmpZ = np.trapz(F_nu_Z_cgs*Ttmp,nu,axis=1)/np.trapz(trans_filters[l],nu_filters[l])
    
    # Here, the part with the log and the zeropoint
    magnitudes_Z_BB[:,l] = -2.5*np.log10(FtmpZ/zeropoint)

# Tell the log that the absolute magnitudes were calculated assuming blackbodies
fid_log = open(log_name,'a')
fid_log.write('ZAMS: Calculated absolute magnitudes assuming blackbody spectra.\n')
fid_log.close()



if only_BB == False:
    # # # #   ZAMS: CALCULATION OF THE ABSOLUTE MAGNITUDES - SPECTRA   # # # #

    # Storage space for the absolute magnitudes from spectra
    magnitudes_spectra_ZAMS = -99.*np.ones([nbr_Z,nbr_filters])

    # Commenting the below to try interpolating over the magnitudes instead

    # Get the interpolated spectra for the ZAMS models
    #ind_Kurucz_possible_ZAMS, Ilambda_interp_SI_ZAMS, Flambda_interp_cgs_ZAMS = interpolate_Kurucz_spectra(10**log_T_ZAMS,log_g_ZAMS,log_R_ZAMS,Teff_Kurucz,log_g_Kurucz,Ilambda_Kurucz_SI,dist_10pc,'log')

    # Creating Fnu to work with the magnitude calculation
    #Fnu_interp_ZAMS = lambda_Kurucz*Flambda_interp_cgs_ZAMS/nu_Kurucz
    #Fnu_interp_ZAMS[np.isnan(Fnu_interp_ZAMS)] = 0.

    # Magnitude calculation
    #for l in range(nbr_filters):
    #    Ttmp = np.interp(lambda_Kurucz, lambda_filters[l], trans_filters[l])
    #    Ftmp = np.trapz(Fnu_interp_ZAMS*Ttmp,nu_Kurucz,axis=1)/np.trapz(trans_filters[l],nu_filters[l])
    #    magnitudes_spectra_ZAMS[ind_Kurucz_possible_ZAMS,l] = -2.5*np.log10(Ftmp/zeropoint)
    
    # Tell the log
    #fid_log = open(log_name,'a')
    #fid_log.write('ZAMS: Interpolated the ZAMS spectra and calculated absolute magnitudes.\n')
    #fid_log.close()

    # Need to start by finding the correct models to interpolate over
    # Get the "corners" - this only allows interpolation and no extrapolation - requiring four corners in the log g - T space
    ind_c1, ind_c2, ind_c3, ind_c4, f1, f2, ind_Kurucz_possible_ZAMS = GetCorners(10**log_T_ZAMS,log_g_ZAMS,T_Kurucz,log_g_Kurucz)
    
    # Need to extend the shape of the array for each ZAMS model ([ZAMS-model, Kurucz-model, filter])
    Itmp = np.array([integral_Inu]) * np.ones([np.sum(ind_Kurucz_possible_ZAMS),1,1])   # nbr_Z[ind_possible] x nbr_Kurucz x nbr_filters
    # Change shape also for f1 and f2
    f1tmp = np.array([f1]).T * np.ones([1,nbr_filters])   # nbr_Z[ind_possible] x nbr_filters
    f2tmp = np.array([f2]).T * np.ones([1,nbr_filters])
    
    # Interpolate over log g
    int_loggmin = Itmp[ind_c1,:] + f1tmp*(Itmp[ind_c4,:]-Itmp[ind_c1,:]) 
    int_loggmax = Itmp[ind_c2,:] + f1tmp*(Itmp[ind_c3,:]-Itmp[ind_c2,:]) 
    # Interpolate over T
    integral_interp = int_loggmin + f2tmp*(int_loggmax-int_loggmin)   # this should be nbr_Z[ind_possible] x nbr_filters

    # Calculation of the absolute magnitudes
    tmp_param = np.pi*(((10**log_R_ZAMS[ind_Kurucz_possible_ZAMS])*RSun_SI/dist_10pc)**2.)  # no unit
    Ftmp = (np.array([tmp_param]).T * np.ones([1,nbr_filters])) * integral_interp    # erg s^-1 cm^-2 Hz^-1
    magnitudes_spectra_ZAMS[ind_Kurucz_possible_ZAMS,:] = -2.5*np.log10(Ftmp/zeropoint)

    # Tell the log
    fid_log = open(log_name,'a')
    fid_log.write('ZAMS: Interpolated (LINEAR-LINEAR in log g and T) the ZAMS absolute magnitudes.\n')
    fid_log.close()



# # # #   ZAMS: OUTPUT FILE   # # # #

# If only the blackbodies are computed
if only_BB:
    # This should contain both the blackbody magnitudes 
    filename_ZAMS = 'ZAMS_Z'+str(Z)+'_BB.txt'
    fid = open(filename_ZAMS,'w')
    fid.write('# This file contains the absolute magnitudes for the ZAMS using blackbody assumptions.\n')
    fid.write('# The ZAMS is based on MESA single star models with Z='+str(Z)+'\n')
    header = '# Minit'
    for k in range(nbr_filters):
        header = header+'\t'+filters[k]+'_BB'
    header = header+'\n'
    fid.write(header)
    for i in range(nbr_Z):
        w_str = str(minit_ZAMS[i])
        for k in range(nbr_filters):
            w_str = w_str+'\t'+str(magnitudes_Z_BB[i,k])           
        w_str = w_str+'\n'
        fid.write(w_str)
    fid.close()


# When also spectra are included
else:
    # This should contain both the blackbody magnitudes and the magnitudes from the spectra
    filename_ZAMS = 'ZAMS_Z'+str(Z)+'.txt'
    fid = open(filename_ZAMS,'w')
    fid.write('# This file contains the absolute magnitudes for the ZAMS using blackbody assumptions and interpolated spectra.\n')
    fid.write('# The ZAMS is based on MESA single star models with Z='+str(Z)+'\n')
    fid.write('# I have used Kurucz models from the grid'+loc_Kurucz+' to model the spectra of the ZAMS.\n')
    header = '# Minit'
    for k in range(nbr_filters):
        header = header+'\t'+filters[k]+'_BB \t '+filters[k]+'_spec'
    header = header+'\n'
    fid.write(header)
    for i in range(nbr_Z):
        w_str = str(minit_ZAMS[i])
        for k in range(nbr_filters):
            w_str = w_str+'\t'+str(magnitudes_Z_BB[i,k])+'\t'+str(magnitudes_spectra_ZAMS[i,k])            
        w_str = w_str+'\n'
        fid.write(w_str)
    fid.close()

# Tell the log
fid_log = open(log_name,'a')
fid_log.write('ZAMS: Wrote a file with the absolute magnitudes for the ZAMS.\n')
fid_log.close()






# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                                                             #
#     DATA FROM MOSS 1                                                        #
#                                                                             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# # # #   READ THE RESULTS FROM:  MOSS part 1   # # # # 
filename_data1 = 'data1_'+run_name+'.txt'
data = np.genfromtxt(filename_data1,dtype='str')

# I am commenting out the parameters we do not need for producing the photometry
star_ID = data[:,0].astype(int)
star_state = data[:,1].astype(int)
evolution = data[:,3]
#minit = data[:,4].astype(float)
m2init = data[:,5]
binary = m2init != '-'
#m1init = minit[binary]
#m2init = m2init[binary].astype(float)
star_state_m1 = star_state[binary]
star_state_m2 = data[:,2][binary].astype(float)
#Pinit = data[:,6][binary].astype(float)
mcurrent = data[:,7].astype(float)
m1current = mcurrent[binary]
#m2current = data[:,8][binary].astype(float)
#Pcurrent = data[:,9][binary].astype(float)
#actual_age = data[:,10].astype(float)
#apparent_age = data[:,11].astype(float)
#apparent_age_m1 = apparent_age[binary]
apparent_age_m2 = data[:,12][binary].astype(float)
log_Tstar = data[:,13].astype(float)
log_Tstar_m1 = log_Tstar[binary]
log_Tstar_m2 = data[:,14][binary].astype(float)
log_g = data[:,15].astype(float)
log_g_m1 = log_g[binary]
log_g_m2 = data[:,16][binary].astype(float)
log_R = data[:,17].astype(float)
#log_R_m1 = log_R[binary]
log_R_m2 = data[:,18][binary].astype(float)
#log_L = data[:,19].astype(float)
#log_L_m1 = log_L[binary]
#log_L_m2 = data[:,20][binary].astype(float)
#log_abs_mdot = data[:,21].astype(float)
#log_abs_mdot_m1 = log_abs_mdot[binary]
#log_abs_mdot_m2 = data[:,22][binary].astype(float)
#XHs = data[:,23].astype(float)
#XHs_m1 = XHs[binary]
#XHs_m2 = data[:,24][binary].astype(float)
#XHes = data[:,25].astype(float)
#XHes_m1 = XHs[binary]
#XHes_m2 = data[:,26][binary].astype(float)

# Delete this enormous array
del data

# Tell the log that the data was read.
fid_log = open(log_name,'a')
fid_log.write('Read '+filename_data1+' with the data from MOSS, part 1\n')
fid_log.close()

# The single stars in this population 
single = binary == False    # Currently just used for a figure

# The number of stars
nbr_stars = len(star_ID)
nbr_binaries = np.sum(binary)

# Evolution of each system (might be future)
evolution_bin = evolution[binary]

# Identify the stripped stars
ind_strip = [False]*nbr_binaries
ind_caseA = [False]*nbr_binaries
ind_caseB = [False]*nbr_binaries
ind_caseB_CEE = [False]*nbr_binaries

for i in range(nbr_binaries):
    # Locate the stripped stars
    if star_state_m1[i] == 2:
        ind_strip[i] = True
        if evolution_bin[i] == 'strip_RLOF_MS':
            ind_caseA[i] = True
        elif evolution_bin[i] == 'strip_RLOF_HG':
            ind_caseB[i] = True
        elif evolution_bin[i] == 'strip_CEE_HG':
            ind_caseB_CEE[i] = True
            
                        
# Stellar types in the population

# primaries and singles (also including mergers)
ind_MS_m = (star_state == 1)+(star_state == 51)
ind_strip_m = np.abs(star_state) == 2
ind_pMS_m = (star_state == 3)+(star_state == 53)
ind_CO_m = (star_state == 4)+(star_state == 54)

# Primaries
ind_MS_m1 = ind_MS_m[binary]
ind_strip_m1 = ind_strip_m[binary]
ind_pMS_m1 = ind_pMS_m[binary]
ind_CO_m1 = ind_CO_m[binary]

ind_strip_tl_m = star_state == -2   # These are too large for their orbit (too young, interpolation fails)
ind_strip_tl_m1 = ind_strip_tl_m[binary]


# Secondaries
ind_MS_m2 = np.abs(star_state_m2) == 1
ind_strip_m2 = np.abs(star_state_m2) == 2
ind_pMS_m2 = np.abs(star_state_m2) == 3
ind_CO_m2 = np.abs(star_state_m2) == 4

ind_overfill_m2 = (star_state_m2 < 0)
#ind_m2_alive = (star_state_m2 != 0)*(star_state_m2 != 4)*(ind_overfill_m2 == False)*(apparent_age_m2 > 0.)
ind_m2_alive = (star_state_m2 != 0)*(np.abs(star_state_m2) != 4)*(apparent_age_m2 > 0.)

# Tell the log that the data was read.
fid_log = open(log_name,'a')
fid_log.write('Sorted stellar types\n')
fid_log.close()


if save_figs:    
    # -- VERIFICATION PLOT -- 
    ms=8

    ylim = [-0.3,6]
    xlim = [3.5,5.1]

    ww = 20
    hh = 6
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(ww,hh))

    for ax in (ax2,ax3):
        # TLUSTY O-star model grid
        #ax.plot(np.log10(T_TLUSTY_O02),logg_TLUSTY_O02,'o',markerfacecolor='none',markeredgecolor=[.8]*3)

        # TLUSTY B-star model grid
        #ax.plot(np.log10(T_TLUSTY_B06),logg_TLUSTY_B06,'o',markerfacecolor='none',markeredgecolor=[.4]*3)

        # BaSeL dwarfs
        #ax.plot(np.log10(T_BaSeL_dwarfs),logg_BaSeL_dwarfs,'xk')
        #ax.plot(np.log10(T_BaSeL_giants),logg_BaSeL_giants,'x',color=[.5]*3)

        if only_BB == False:
            # Kurucz model grid
            ax.plot(np.log10(T_Kurucz),logg_Kurucz,'+k',label='Kurucz grid')



    # ax1 is for stripped stars
    ax1.plot(log_Tstar_m1[ind_strip_m1],log_g_m1[ind_strip_m1],'.',color=[.6,.3,.6],ms=ms, label='All')
    #ax1.plot(log_Tstar_m1[ind_strip_m1][ind_interp_strip], log_g_m1[ind_strip_m1][ind_interp_strip], '.', color=[1,0,0],ms=ms,
    #        label='Spectral interpolation')
    if only_BB == False:
        ax1.plot(np.log10(Tstar_stripped),logg_stripped,'.',color='k',ms=ms,label='Model grid')
    ax1.legend(loc=0,fontsize=0.6*fsize,edgecolor='none')

    # ax2 is for accretors
    # The secondary stars to stripped stars
    ind_tmp = ind_m2_alive*ind_strip_m1
    ax2.plot(log_Tstar_m2[ind_tmp],log_g_m2[ind_tmp],'.',ms=ms,label='All')
    ax2.plot(log_Tstar_m2[ind_tmp*ind_MS_m2],log_g_m2[ind_tmp*ind_MS_m2],'.m',ms=ms,label='MS')
    ax2.plot(log_Tstar_m2[ind_tmp*ind_pMS_m2],log_g_m2[ind_tmp*ind_pMS_m2],'.',color='olive',ms=ms,label='pMS')
    ax2.plot(log_Tstar_m2[ind_tmp*(star_state_m2 == 0)],log_g_m2[ind_tmp*(star_state_m2 == 0)],'.',color='blue',
             ms=ms,label='CO')
    ax2.legend(loc=0,fontsize=0.6*fsize,edgecolor='none')


    # ax3 is for the remaining stars
    ax3.plot(log_Tstar[single],log_g[single],'.k',ms=ms)   # Single stars from beginning
    ax3.plot(log_Tstar_m1[star_state_m1==1], log_g_m1[star_state_m1==1],'.k',ms=ms)   # MS primaries that didn't interact yet
    ax3.plot(log_Tstar_m1[star_state_m1==3], log_g_m1[star_state_m1==3],'.k',ms=ms)   # post-MS primaries
    ax3.plot(log_Tstar_m1[star_state_m1==51], log_g_m1[star_state_m1==51],'.k',ms=ms)  # mergers on MS
    ax3.plot(log_Tstar_m1[star_state_m1==53], log_g_m1[star_state_m1==53],'.k',ms=ms)  # mergers on post MS
    ax3.plot(log_Tstar_m2[star_state_m2==1], log_g_m2[star_state_m2==1],'.k',ms=ms)    # secondaries on MS (not yet inter.)
    ax3.plot(log_Tstar_m2[star_state_m2==3], log_g_m2[star_state_m2==3],'.k',ms=ms)    # secondaries on post MS (not yet inter.)


    for ax in  (ax1, ax2, ax3):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_xlabel('$\\log_{10} T_{\\mathrm{eff}}/\mathrm{K}$')
        ax.set_ylabel('$\\log_{10} g$')

    ax1.set_title('Stripped stars')
    ax2.set_title('Secondaries to stripped stars')
    ax3.set_title('Singles, mergers, nyi binaries')
    fig.savefig(plots_dir+'/spectral_grids_parameter_space.png',format='png',bbox_inches='tight',pad_inches=0.1)
    plt.close(fig)
    # -- -- -- -- -- 

# Tell the log
fid_log = open(log_name,'a')
fid_log.write('Saved a spectral model grid parameter space coverage figure.\n')
fid_log.close()



# # # #   STARTING A LOOP - HAPPENING IF THERE ARE LOTS OF STARS   # # # #
# 
# How to split this? Simplest way would be to take the single+primary array
# first and then the secondary array. Let's see how this works out.. 

# Storage space for the blackbody magnitudes
magnitudes_BB = np.zeros([nbr_stars,nbr_filters])
magnitudes_m2_BB = np.zeros([nbr_binaries,nbr_filters])

if only_BB == False:
    # Storage space for magnitudes
    magnitudes_m_spec = -99.*np.ones([nbr_stars,nbr_filters])   
    magnitudes_m1_spec = -99.*np.ones([nbr_binaries,nbr_filters])
    magnitudes_m2_spec = -99.*np.ones([nbr_binaries,nbr_filters]) 


# Calculate how many loops that are needed
nn = 100000.
num_big_loops = np.ceil(nbr_stars/nn) 
#num_big_loops_bin = np.ceil(nbr_binaries/nn)

# Tell the log how many loops are needed
fid_log = open(log_name,'a')
fid_log.write('Need to loop '+str(int(num_big_loops))+' times...\n')
fid_log.close()

indices_1 = np.arange(nbr_stars)
indices_2 = np.arange(nbr_binaries)

# Here starts the loop!
for iii in range(int(num_big_loops)):
    
    # Construct the indices for the arrays
    ind_1 = (indices_1 >= (iii*nn))*(indices_1 < ((iii+1.)*nn))
    ind_2 = (indices_2 >= (iii*nn))*(indices_2 < ((iii+1.)*nn))
    loop_2 = np.sum(ind_2) > 0.  # Switch for whether the binaries need to be included
    
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                                                                             #
    #     BLACKBODY RADIATION                                                     #
    #                                                                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    # # # # For the stars in the population
    # Going to get the magnitudes for just stars that I have a reasonable model for
    # All primaries and single stars are alive, these should be fine to use

    # Calculate the blackbody intensities for the singles and primaries
    [ll,TT] = np.meshgrid(lambda_SI,10**log_Tstar[ind_1])  #  Temperature in K
    [ll,RR] = np.meshgrid(lambda_SI,(10**log_R[ind_1])*u.R_sun.to(u.m))      # Radius in meters
    B_lambda = (2.*h_SI*(c_SI**2.)/(ll**5.))/(np.exp(h_SI*c_SI/(ll*kB_SI*TT))-1.)

    # Translate that into flux at a distance of 10 pc (handy for absolute magnitudes)
    F_lambda_SI = np.pi*B_lambda*((RR/dist_10pc)**2.)
    
    # Put this in cgs units
    F_lambda_cgs = F_lambda_SI/1e7
    
    # And calculate the Fnu
    F_nu_cgs = lambda_AA*F_lambda_cgs/nu
    
    
    # Tell the log that blackbodies were created
    fid_log = open(log_name,'a')
    fid_log.write('Created blackbodies for the stars in single+primary array, loop number '+str(iii)+'.\n')
    fid_log.close()

    # Do the same for the secondaries?
    if loop_2:

        [ll2,TT2] = np.meshgrid(lambda_SI,10**log_Tstar_m2[ind_2*ind_m2_alive])  #  Temperature in K
        [ll2,RR2] = np.meshgrid(lambda_SI,(10**log_R_m2[ind_2*ind_m2_alive])*u.R_sun.to(u.m))      # Radius in meters
        B_lambda_m2 = (2.*h_SI*(c_SI**2.)/(ll2**5.))/(np.exp(h_SI*c_SI/(ll2*kB_SI*TT2))-1.)
        F_lambda_m2_SI = np.pi*B_lambda_m2*((RR2/dist_10pc)**2.)
        F_lambda_m2_cgs = F_lambda_m2_SI/1e7
        F_nu_m2_cgs = lambda_AA*F_lambda_m2_cgs/nu

        fid_log = open(log_name,'a')
        fid_log.write('Created blackbodies for the secondaries too.\n')
        fid_log.close()


        
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                                                                             #
    #     Calculate the magnitudes -- Blackbodies                                 #
    #                                                                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    # # # # CALCULATION OF THE ABSOLUTE MAGNITUDES # # # #

    # Loop over the filters and calculate the magntiudes for the stars for each filter
    for l in range(nbr_filters):

        # Calculate the magnitude in the current band for all stars
        # This is the int(Fnu*T(nu))/int(T(nu)) part
        # More accurate with the wavelength array of the spectra below
        Ttmp = np.interp(lambda_AA, lambda_filters[l], trans_filters[l])
        Ftmp = np.trapz(F_nu_cgs*Ttmp,nu,axis=1)/np.trapz(trans_filters[l],nu_filters[l])
        
        # Here, the part with the log and the zeropoint
        magnitudes_BB[ind_1,l] = -2.5*np.log10(Ftmp/zeropoint)
        
        if loop_2:
            Ftmp2 = np.trapz(F_nu_m2_cgs*Ttmp,nu,axis=1)/np.trapz(trans_filters[l],nu_filters[l])
            magnitudes_m2_BB[ind_2*ind_m2_alive,l] = -2.5*np.log10(Ftmp2/zeropoint)


    # Tell the log that the absolute magnitudes were calculated assuming blackbodies
    fid_log = open(log_name,'a')
    fid_log.write('Calculated absolute magnitudes assuming blackbody spectra.\n')
    fid_log.close()



    # In case the spectra are also used
    if only_BB == False:
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        #                                                                             #
        #     Spectral model interpolation                                            #
        #                                                                             #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             


        # I will just interpolate the spectra that are close enough to the original models
        # Also, the grid varies over stripped star mass, I think this is the best way to go for the interpolation

        if loop_2:
            # (1) choose the stripped stars that should use the stripped star model spectra
            # Find the closest model to each stripped star, in terms of log Teff and log g
            [Tm,Ts] = np.meshgrid(np.log10(Tstar_stripped),log_Tstar_m1[ind_2*ind_strip_m1])
            [gm,gs] = np.meshgrid(logg_stripped,log_g_m1[ind_2*ind_strip_m1])
            delta_logg = gm-gs
            delta_logT = Tm-Ts
            diff = np.sqrt((delta_logg**2.) + (delta_logT**2.))

            ind = diff == ((np.min(diff,axis=1)*np.ones([len(Tstar_stripped),1])).T)
            diff = diff[ind]
            delta_logg = delta_logg[ind]
            delta_logT = delta_logT[ind]

            ind_interp_strip = (delta_logg < 0.5)*(delta_logT < 0.15)



        # # # KURUCZ INTERPOLATION

        # Start with the MS and pMS stars (Kurucz)
        ind_iK_m = (ind_MS_m+ind_pMS_m)>0    # These are MS and pMS stars that will be represented by Kurucz models 
        Tstar_iK_m = 10**log_Tstar[ind_1*ind_iK_m]
        log_g_iK_m = log_g[ind_1*ind_iK_m]
        log_R_iK_m = log_R[ind_1*ind_iK_m]
        tot_iK_m = np.sum(ind_1*ind_iK_m)

        # Get the number of loops needed to get all the spectra interpolated  -- CAN MAYBE REMOVE THIS LOOP
        #num_loops = int(np.ceil(tot_iK_m/max_num))

        # Tell the log
        #fid_log = open(log_name,'a')
        #fid_log.write('Need to loop '+str(num_loops)+' times and there is '+str(tot_iK_m)+' stars in this big loop\n')
        #fid_log.close()

        # Just for keeping track of which stars we calculate the magnitudes for here
        magnitudes_iK_m_spec = -99.*np.ones([tot_iK_m,nbr_filters])

        ## Start looping ...
        #for i in range(num_loops):

        #    # Do the interpolation for a batch of parameters
        #    i1 = int(i*max_num)
        #    i2 = int((i+1)*max_num)
        #    if i == (num_loops-1):
        #        i2 = int(copy.copy(tot_iK_m))
        
        # Need to start by finding the correct models to interpolate over
        # Get the "corners" - this only allows interpolation and no extrapolation - requiring four corners in the log g - T space
        #ind_c1, ind_c2, ind_c3, ind_c4, f1, f2, ind_Kurucz_possible = GetCorners(Tstar_iK_m,log_g_iK_m,T_Kurucz,log_g_Kurucz)

        # Loop over the filters -- REMOVE THIS LOOP
        #for l in range(nbr_filters):

        #    # Interpolate over log g
        #    int_loggmin = integral_Inu[ind_c1,l] + f1*(integral_Inu[ind_c4,l]-integral_Inu[ind_c1,l]) 
        #    int_loggmax = integral_Inu[ind_c2,l] + f1*(integral_Inu[ind_c3,l]-integral_Inu[ind_c2,l]) 
        #    # Interpolate over T
        #    integral_interp = int_loggmin + f2*(int_loggmax-int_loggmin)

        #    # Calculation of the absolute magnitudes
        #    Ftmp = (np.pi/1e7)*(((10**log_R_iK_m[ind_Kurucz_possible])*RSun_SI/dist_10pc)**2.) * integral_interp
        #    mtmp = magnitudes_iK_m_spec[:,l]
        #    mtmp[ind_Kurucz_possible] = -2.5*np.log10(Ftmp/zeropoint)
        #    magnitudes_iK_m_spec[:,l] = mtmp

            
        # Need to start by finding the correct models to interpolate over
        # Get the "corners" - this only allows interpolation and no extrapolation - requiring four corners in the log g - T space
        ind_c1, ind_c2, ind_c3, ind_c4, f1, f2, ind_Kurucz_possible = GetCorners(Tstar_iK_m,log_g_iK_m,T_Kurucz,log_g_Kurucz)

        # Need to extend the shape of the array for each ZAMS model ([ZAMS-model, Kurucz-model, filter])
        Itmp = np.array([integral_Inu]) * np.ones([np.sum(ind_Kurucz_possible),1,1])   # nbr_stars[ind_possible] x nbr_Kurucz x nbr_filters
        # Change shape also for f1 and f2
        f1tmp = np.array([f1]).T * np.ones([1,nbr_filters])   # nbr_stars[ind_possible] x nbr_filters
        f2tmp = np.array([f2]).T * np.ones([1,nbr_filters])

        # Interpolate over log g
        int_loggmin = Itmp[ind_c1,:] + f1tmp*(Itmp[ind_c4,:]-Itmp[ind_c1,:]) 
        int_loggmax = Itmp[ind_c2,:] + f1tmp*(Itmp[ind_c3,:]-Itmp[ind_c2,:]) 
        # Interpolate over T
        integral_interp = int_loggmin + f2tmp*(int_loggmax-int_loggmin)   # this should be nbr_stars[ind_possible] x nbr_filters

        # Calculation of the absolute magnitudes
        tmp_param = np.pi*(((10**log_R_iK_m[ind_Kurucz_possible])*RSun_SI/dist_10pc)**2.)  # no unit
        Ftmp = (np.array([tmp_param]).T * np.ones([1,nbr_filters])) * integral_interp    # erg s^-1 cm^-2 Hz^-1
        magnitudes_iK_m_spec[ind_Kurucz_possible,:] = -2.5*np.log10(Ftmp/zeropoint)

            
            
    #ind_Kurucz_possible, Ilambda_interp_SI, Flambda_interp_cgs = interpolate_Kurucz_spectra(Tstar_iK_m[i1:i2],log_g_iK_m[i1:i2],log_R_iK_m[i1:i2],Teff_Kurucz,log_g_Kurucz,Ilambda_Kurucz_SI,dist_10pc,'log')

    # Calculate the magnitudes for these stars 
    #Fnu_interp_Kurucz = lambda_Kurucz*Flambda_interp_cgs/nu_Kurucz
    #Fnu_interp_Kurucz[np.isnan(Fnu_interp_Kurucz)] = 0.

    #for l in range(nbr_filters):
        #Ttmp = np.interp(lambda_Kurucz, lambda_filters[l], trans_filters[l])
        #Ftmp = np.trapz(Fnu_interp_Kurucz*Ttmp,nu_Kurucz,axis=1)/np.trapz(trans_filters[l],nu_filters[l])

        # Here, the part with the log and the zeropoint
        #mtmp = magnitudes_iK_m_spec[i1:i2,l]
        #mtmp[ind_Kurucz_possible] = -2.5*np.log10(Ftmp/zeropoint)
        #magnitudes_iK_m_spec[i1:i2,l] = mtmp


        # Tell the log
        #fid_log = open(log_name,'a')
        #fid_log.write('Inside of loop for mag-calculation (m), at '+str(i+1)+'...\n')
        #fid_log.close()

        # Save the absolute magnitudes calculated from the interpolated spectra
        magnitudes_m_spec[ind_1*ind_iK_m] = magnitudes_iK_m_spec

        # Tell the log
        fid_log = open(log_name,'a')
        fid_log.write('Saved the magnitudes for MS and pMS primaries and singles (too much data to save the spectra).\n')
        fid_log.close()



        if loop_2:
            # Go to secondaries (MS-stars and pMS-stars)
            ind_iK_m2 = (ind_MS_m2+ind_pMS_m2) > 0    # These are MS and pMS stars that will be represented by Kurucz models
            Tstar_iK_m2 = 10**log_Tstar_m2[ind_2*ind_iK_m2]
            log_g_iK_m2 = log_g_m2[ind_2*ind_iK_m2]
            log_R_iK_m2 = log_R_m2[ind_2*ind_iK_m2]
            tot_iK_m2 = np.sum(ind_2*ind_iK_m2)

            # Get the number of loops needed to get all the spectra interpolated
            #num_loops = int(np.ceil(tot_iK_m2/max_num))

            # Just for keeping track of which stars we calculate the magnitudes for here
            magnitudes_iK_m2_spec = -99.*np.ones([tot_iK_m2,nbr_filters])

            # Start looping ...
            #for i in range(num_loops):

            # Do the interpolation for a batch of parameters
            #i1 = int(i*max_num)
            #i2 = int((i+1)*max_num)
            #if i == (num_loops-1):
            #    i2 = int(copy.copy(tot_iK_m2))
            #ind_Kurucz_possible, Ilambda_interp_SI, Flambda_interp_cgs = interpolate_Kurucz_spectra(Tstar_iK_m2[i1:i2],log_g_iK_m2[i1:i2],log_R_iK_m2[i1:i2],Teff_Kurucz,log_g_Kurucz,Ilambda_Kurucz_SI,dist_10pc,'log')

            # Calculate the magnitudes for these stars 
            #Fnu_interp_Kurucz = lambda_Kurucz*Flambda_interp_cgs/nu_Kurucz
            #Fnu_interp_Kurucz[np.isnan(Fnu_interp_Kurucz)] = 0.

            # Need to start by finding the correct models to interpolate over
            # Get the "corners" - this only allows interpolation and no extrapolation - requiring four corners in the log g - T space
            ind_c1, ind_c2, ind_c3, ind_c4, f1, f2, ind_Kurucz_possible = GetCorners(Tstar_iK_m2,log_g_iK_m2,T_Kurucz,log_g_Kurucz)

            # Need to extend the shape of the array for each ZAMS model ([ZAMS-model, Kurucz-model, filter])
            Itmp = np.array([integral_Inu]) * np.ones([np.sum(ind_Kurucz_possible),1,1])   # nbr_stars[ind_possible] x nbr_Kurucz x nbr_filters
            # Change shape also for f1 and f2
            f1tmp = np.array([f1]).T * np.ones([1,nbr_filters])   # nbr_stars[ind_possible] x nbr_filters
            f2tmp = np.array([f2]).T * np.ones([1,nbr_filters])

            # Interpolate over log g
            int_loggmin = Itmp[ind_c1,:] + f1tmp*(Itmp[ind_c4,:]-Itmp[ind_c1,:]) 
            int_loggmax = Itmp[ind_c2,:] + f1tmp*(Itmp[ind_c3,:]-Itmp[ind_c2,:]) 
            # Interpolate over T
            integral_interp = int_loggmin + f2tmp*(int_loggmax-int_loggmin)   # this should be nbr_stars[ind_possible] x nbr_filters

            # Calculation of the absolute magnitudes
            tmp_param = np.pi*(((10**log_R_iK_m2[ind_Kurucz_possible])*RSun_SI/dist_10pc)**2.)  # no unit
            Ftmp = (np.array([tmp_param]).T * np.ones([1,nbr_filters])) * integral_interp    # erg s^-1 cm^-2 Hz^-1
            magnitudes_iK_m2_spec[ind_Kurucz_possible,:] = -2.5*np.log10(Ftmp/zeropoint)


            # Tell the log
            #fid_log = open(log_name,'a')
            #fid_log.write('Inside of loop for mag-calculation (m2), at '+str(i+1)+'...\n')
            #fid_log.close()

            magnitudes_m2_spec[ind_2*ind_iK_m2] = magnitudes_iK_m2_spec


            # Tell the log
            fid_log = open(log_name,'a')
            fid_log.write('Saved the magnitudes for MS and pMS secondaries (too much data to save the spectra).\n')
            fid_log.close()




            # # # # Interpolation for stripped stars # # # # 
            # Now, I can interpolate the spectra - updating to interpolate the magnitudes
            mstrip_interp = m1current[ind_2*ind_strip_m1][ind_interp_strip]
            #Flambda_strip_interp = np.zeros([len(mstrip_interp),len(lambda_Kurucz)])
            abs_mag_strip_interp_spec = np.zeros([len(mstrip_interp),nbr_filters])

            # I will loop over the models, they are not that many
            for i in range(nbr_stripped_models-1):

                # Which stars are included here? 
                if i < (nbr_stripped_models-1):
                    ind_i = ((mstrip_interp > mass_stripped[i])*(mstrip_interp < mass_stripped[i+1]))

                elif i == (nbr_stripped_models-1):
                    ind_i = mstrip_interp > mass_stripped[i]

                # How to interpolate
                # Upgrading to magnitude interpolation (Ylva, 17 August 2020)
                if interp_scheme == 'linlin':
                    k = (mstrip_interp[ind_i] - mass_stripped[i])/(mass_stripped[i+1]-mass_stripped[i])
                    #[F1,kk] = np.meshgrid(Flambda_stripped_models_coarse[i],k)
                    #[F2,kk] = np.meshgrid(Flambda_stripped_models_coarse[i+1],k)
                    #Flambda_strip_interp[ind_i,:] = F1+kk*(F2-F1)        
                    [am1,kk] = np.meshgrid(abs_mag_stripped_grid[i,:],k)
                    [am2,kk] = np.meshgrid(abs_mag_stripped_grid[i+1,:],k)
                    abs_mag_strip_interp_spec[ind_i,:] = am1 + kk*(am2-am1)   # Linear interpolation in mass and absolute magnitude

                # I think this will not work with absolute magnitudes (they are negative sometimes) 
                elif interp_scheme == 'linlog':
                    k = (mstrip_interp[ind_i] - mass_stripped[i])/(mass_stripped[i+1]-mass_stripped[i])
                    #[F1,kk] = np.meshgrid(np.log10(Flambda_stripped_models_coarse[i]),k)
                    #[F2,kk] = np.meshgrid(np.log10(Flambda_stripped_models_coarse[i+1]),k)
                    #Flambda_strip_interp[ind_i,:] = 10**(F1+kk*(F2-F1))
                    [am1,kk] = np.meshgrid(np.log10(abs_mag_stripped_grid[i,:]),k)
                    [am2,kk] = np.meshgrid(np.log10(abs_mag_stripped_grid[i+1,:]),k)
                    abs_mag_strip_interp_spec[ind_i,:] = 10**(am1+kk*(am2-am1))

                # This is linear in absolute magnitude and log in mass
                elif interp_scheme == 'loglin':
                    k = (np.log10(mstrip_interp[ind_i]) - np.log10(mass_stripped[i]))/(np.log10(mass_stripped[i+1])-np.log10(mass_stripped[i]))
                    [am1,kk] = np.meshgrid(abs_mag_stripped_grid[i,:],k)
                    [am2,kk] = np.meshgrid(abs_mag_stripped_grid[i+1,:],k)
                    abs_mag_strip_interp_spec[ind_i,:] = am1 + kk*(am2-am1)   # Logarithmic interpolation in mass and linear in absolute magnitude

                # This will not work now with absolute magnitudes (they are sometimes negative)
                elif interp_scheme == 'loglog':
                    k = (np.log10(mstrip_interp[ind_i]) - np.log10(mass_stripped[i]))/(np.log10(mass_stripped[i+1])-np.log10(mass_stripped[i]))
                    #[F1,kk] = np.meshgrid(np.log10(Flambda_stripped_models_coarse[i]),k)
                    #[F2,kk] = np.meshgrid(np.log10(Flambda_stripped_models_coarse[i+1]),k)
                    #Flambda_strip_interp[ind_i,:] = 10**(F1+kk*(F2-F1))
                    [am1,kk] = np.meshgrid(np.log10(abs_mag_stripped_grid[i,:]),k)
                    [am2,kk] = np.meshgrid(np.log10(abs_mag_stripped_grid[i+1,:]),k)
                    abs_mag_strip_interp_spec[ind_i,:] = 10**(am1+kk*(am2-am1))

            # Tell the log how it is interpolated
            if interp_scheme == 'linlin':
                fid_log = open(log_name,'a')
                #fid_log.write('This is an interpolation for the stripped star spectra linearly over mass and linearly over the Flambda.\n')
                fid_log.write('This is an interpolation for the stripped star spectra linearly over mass and linearly over the absolute magnitude.\n')
                fid_log.close()
            elif interp_scheme == 'linlog':
                fid_log = open(log_name,'a')
                fid_log.write('This is an interpolation for the stripped star spectra linearly over mass and logarithmically over absolute magnitude.\n')
                fid_log.close()
            elif interp_scheme == 'loglin':
                fid_log = open(log_name,'a')
                fid_log.write('This is an interpolation for the stripped star spectra logarithmically over mass and linearly over absolute magnitude.\n')
                fid_log.close()
            elif interp_scheme == 'loglog':
                fid_log = open(log_name,'a')
                fid_log.write('This is an interpolation for the stripped star spectra logarithmically over mass and logarithmically over absolute magnitude.\n')
                fid_log.close()


            #Flambda_strip_interp = Flambda_strip_interp*((1e3*pc_SI/dist_10pc)**2.)

            # Tell the log
            fid_log = open(log_name,'a')
            #fid_log.write('Interpolated the stripped star SEDs and put them at 10 pc distance (for absolute magnitudes)\n')
            fid_log.write('Interpolated the stripped star absolute magnitudes...\n')
            fid_log.close()


            # # # CALCULATE THE MAGNITUDES (stripped stars + ZAMS)
            # Need to change the flux units
            #Fnu_strip_cgs = lambda_Kurucz*Flambda_strip_interp/nu_Kurucz

            # Storage space for the magnitudes
            #magnitudes_spectra_strip = np.zeros([np.sum(ind_interp_strip),nbr_filters])


            # Loop over the filters and calculate the magntiudes for the stars for each filter
            #for l in range(nbr_filters):

                # Calculate the magnitude in the current band for all stars
            #    Ttmp = np.interp(lambda_Kurucz, lambda_filters[l], trans_filters[l])
            #    Tsum = np.trapz(trans_filters[l],nu_filters[l])
            #    Ftmp = np.trapz(Fnu_strip_cgs*Ttmp,nu_Kurucz,axis=1)/Tsum
            #    magnitudes_spectra_strip[:,l] = -2.5*np.log10(Ftmp/zeropoint)

            # Update the general magnitudes matrix
            mtmp = magnitudes_m1_spec[ind_2*ind_strip_m1,:]
            mtmp[ind_interp_strip,:] = abs_mag_strip_interp_spec
            magnitudes_m1_spec[ind_2*ind_strip_m1,:] = mtmp 
            magnitudes_m_spec[binary,:] = magnitudes_m1_spec

            # Tell the log
            fid_log = open(log_name,'a')
            fid_log.write('Put the absolute magnitudes for the interpolated stripped stars in the large arrays\n')
            fid_log.close()

    # Finished with this loop
    fid_log = open(log_name,'a')
    fid_log.write('Finished loop '+str(int(iii))+'......\n')
    fid_log.close()

        

# Put together everything into arrays for binaries
        
# # # # BLACKBODIES 
# The primaries in the binaries
magnitudes_m1_BB = magnitudes_BB[binary,:]
# The total magnitudes of the binaries
magnitudes_bin_BB = copy.copy(magnitudes_m1_BB)
magnitudes_bin_BB[ind_m2_alive] = -2.5*np.log10(10**(-0.4*magnitudes_m1_BB[ind_m2_alive]) + 10**(-0.4*magnitudes_m2_BB[ind_m2_alive]))

if only_BB == False:
    # # # # SPECTRA
    # Create the ones for only primaries
    magnitudes_m1_spec = magnitudes_m_spec[binary,:]
    # And calculate the joint magnitudes
    magnitudes_bin_spec = -2.5*np.log10(10**(-0.4*magnitudes_m1_spec) + 10**(-0.4*magnitudes_m2_spec))
    ind1 = magnitudes_m1_spec == -99.
    magnitudes_bin_spec[ind1] = copy.copy(magnitudes_m2_spec[ind1])
    ind2 = magnitudes_m2_spec == -99.
    magnitudes_bin_spec[ind2] = copy.copy(magnitudes_m1_spec[ind2])
    
# Tell the log
fid_log = open(log_name,'a')
fid_log.write('Calculated the joint magnitudes for the binaries.\n')
fid_log.close()



# # # OUTPUT # # #

# # # POPULATION MAGNITUDES # # # 

# This is helping me understand what I write down below in the file
# For blackbody
mag_m2_BB_tmp = -99.*np.ones(magnitudes_BB.shape)
mtmp = mag_m2_BB_tmp[binary]
mtmp[ind_m2_alive] = copy.copy(magnitudes_m2_BB[ind_m2_alive])
mag_m2_BB_tmp[binary] = mtmp
if only_BB == False:
    # For spectrum
    mag_m2_spec_tmp = -99.*np.ones(magnitudes_BB.shape)
    mtmp = mag_m2_spec_tmp[binary]
    mtmp[ind_m2_alive] = magnitudes_m2_spec[ind_m2_alive]
    mag_m2_spec_tmp[binary] = mtmp

# Name for the data2 file
filename = 'data2_'+run_name+'.txt'

# Write a description
fid = open(filename,'w')
fid.write('# \n')
fid.write('#       Data from spectral population synthesis for '+run_name+'\n')
fid.write('# \n')
date = str(datetime.datetime.now()).split(' ')[0]
fid.write('# By Ylva Gotberg, '+date+'\n')
fid.write('#\n')

fid.write('# This contains the colors of the stars presented in the population synthesis in the file data1_'+run_name+'.txt\n')
if only_BB:
    fid.write('# There are colors from blackbody assumptions. \n')
else:
    fid.write('# There are colors both from blackbody assumptions and from spectra. \n')
    fid.write('# The spectra are interpolated using Kurucz models and models for stripped stars.\n')

write_str = '# Star_ID '
for i in range(nbr_filters):
    if only_BB:
        write_str = write_str+' \t '+filters[i]+'_BB_m1 \t '+filters[i]+'_BB_m2  '
    else:
        write_str = write_str+' \t '+filters[i]+'_BB_m1 \t '+filters[i]+'_BB_m2 \t'+filters[i]+'_spec_m1 \t '+filters[i]+'_spec_m2  '
write_str = write_str+'\n'
fid.write(write_str)

fac = 1000.

for i in range(len(star_ID)):
    write_str = str(star_ID[i])
    
    for k in range(nbr_filters):
        if only_BB:
            write_str = write_str+'\t'+str(np.round(fac*magnitudes_BB[i,k])/fac)+'\t '+str(np.round(fac*mag_m2_BB_tmp[i,k])/fac)+' '
        else:
            write_str = write_str+'\t'+str(np.round(fac*magnitudes_BB[i,k])/fac)+'\t '+str(np.round(fac*mag_m2_BB_tmp[i,k])/fac)+' \t' +str(np.round(fac*magnitudes_m_spec[i,k])/fac)+'\t '+str(np.round(fac*mag_m2_spec_tmp[i,k])/fac)+' '
                
    fid.write(write_str+'\n')
fid.close()
