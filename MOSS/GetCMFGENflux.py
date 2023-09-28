# -*- coding: utf-8 -*-
import numpy as np
from sympy import cos, atan, sqrt, N
from constants import constants

# Give as input a string with the location of the model
def GetCMFGENflux(model_loc,file_flux):
    """
    
    Parameters:
    -----------
    model_loc : string
        The location of the model
    file_flux : string
        The file with the fluxes

    Returns:
    --------
    Flux_lambda_SI_full : array
        The fluxes in W m^-1
    lambda_SI : array
        The wavelengths in meter
    Flux_nu_SI_full : array
        The fluxes in W Hz^-1
    freq_SI : array
        The frequencies in Hz
    Flambda_cgs : array
        The fluxes in erg s^-1 cm^-2 Å^-1
    Fnu_cgs : array
        The fluxes in erg s^-1 cm^-2 Hz^-1

    Notes:
    ------
    This function reads the output files for a given 
    CMFGEN model and returns the wavelengths, frequencies
    and fluxes (Fnu and Flambda). The fluxes are assumed 
    that all emission is seen (no 1 kpc assumption or 
    similar). All returned variables are given in SI 
    units, sort of. Wavelength is in meters, frequency in
    hertz, Flambda in W m^-1 and Fnu in W Hz^-1.

    Author:     Ylva Götberg
    Date:       28/9 - 2015    
    """
    
    cstes = constants()
    c_SI = cstes['c_SI']
    pc_SI = cstes['pc_SI']
    RSun_SI = cstes['RSun_SI']
    
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # FREQUENCY AND FLUX  # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    # Where is the file with fluxes? (Use OBSFLUX if obs_fin is not given)
    filename_fluxes = model_loc + '/OBSFLUX'
    if file_flux != '': filename_fluxes = file_flux

    # Read the file
    f = open(filename_fluxes,'r')
    f_lines = f.readlines()
    nbr_lines = len(f_lines)

    # What to search for
    str_1_freq = 'Continuum'
    str_2_freq = 'Frequencies'
    str_1_flux = 'Observed'
    str_2_flux = 'intensity'
    str_3_flux = '(Janskys)'

    # Get the regions in the file corresponding to the frequencies and fluxes
    # # # This is designed for the OBSFLUX and obs_fin files
    ind_end_flux = []
    for i in range(nbr_lines):
        tmp = f_lines[i].split()
        if len(tmp):
            if (tmp[0] == str_1_freq) and (tmp[1] == str_2_freq):
                ind_begin_freq = i+2
            elif (tmp[0] == str_1_flux) and (tmp[1] == str_2_flux) and (tmp[2] == str_3_flux):
                ind_end_freq = i-3
                ind_begin_flux = i+2
            elif (tmp[0] == 'Mechanical') and (tmp[1] == 'Luminosity'):
                ind_end_flux = i-11
                
    if ind_end_flux == []: ind_end_flux = nbr_lines
        
    # Get the frequencies (10^15 Hz)
    freq = []
    for i in range(len(f_lines[ind_begin_freq:ind_end_freq])):
        tmp = f_lines[ind_begin_freq+i].split()
        for j in range(len(tmp)):
            freq.append(float(tmp[j]))

    # Get all the fluxes (Jy at 1 kpc distance)
    flux = []
    for i in range(len(f_lines[ind_begin_flux:ind_end_flux])):
        tmp = f_lines[ind_begin_flux+i].split()
        for j in range(len(tmp)):
            if 'E' in tmp[j]:
                flux.append(float(tmp[j]))
            elif '-' in tmp[j]:
                tmp[j] = tmp[j][0:5]+'E'+tmp[j][6:]
                flux.append(float(tmp[j]))
            else:
                flux.append(float(tmp[j]))
                            
            
    # If flux and freq have different length, cut to the shortest
    min_len = min(len(flux),len(freq))
    Fnu_Jy = flux[0:min_len]      # This is Fnu at 1 kpc distance [Jy]
    freq = freq[0:min_len]
            
    # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # INTO SI UNITS # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    # Frequency in Hz
    freq_SI = np.array(freq)*1e15

    # Wavelength
    lambda_SI = c_SI/freq_SI      # in meter
    lambda_AA = np.array(lambda_SI)*1e10    # in Ångström

    # Get Flambda at 1 kpc distance in cgs (erg s^-1 cm^-2 Å^-1)
    # nu * Fnu = lambda * Flambda
    Fnu_cgs = 1e-23*np.array(Fnu_Jy)
    Flambda_cgs = freq_SI*Fnu_cgs/lambda_AA
    
    # F_nu in J s^-1 m^-2 Hz^-1
    Fnu_SI = np.array(Fnu_Jy)*1e-26
    
    # F_lambda in J s^-1 m^-3
    Flambda_SI = Fnu_SI*freq_SI/lambda_SI
    
    # F_lambda in J s^-1 m^-1
    Flux_lambda_SI_full = Flambda_SI*4*np.pi*(1e3*pc_SI)**2.0
    
    # F_nu in J s^-1 Hz^-1
    Flux_nu_SI_full = Fnu_SI*4*np.pi*(1e3*pc_SI)**2.0

    return Flux_lambda_SI_full, lambda_SI, Flux_nu_SI_full, freq_SI, Flambda_cgs, Fnu_cgs