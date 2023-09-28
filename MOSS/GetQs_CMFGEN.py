# -*- coding: utf-8 -*-
# Import some packages
import numpy as np
from integrate_BB_star import integrate_BB_star
from GetCMFGENflux import GetCMFGENflux
from constants import constants


def GetQs_CMFGEN(model_loc, Teff):
    """
    
    Parameters:
    -----------
    model_loc : string
        The location of the model
    Teff : float
        The effective temperature of the star

    Returns:
    --------
    B_lambda : array
        The blackbody emission in W m^-1
    Q0_bb : float
        The rate of emitted hydrogen ionising photons from the blackbody
    Q1_bb : float
        The rate of emitted HeI ionising photons from the blackbody
    Q2_bb : float
        The rate of emitted HeII ionising photons from the blackbody
    Q0_s : float
        The rate of emitted hydrogen ionising photons from the spectrum
    Q1_s : float
        The rate of emitted HeI ionising photons from the spectrum
    Q2_s : float
        The rate of emitted HeII ionising photons from the spectrum
    
    Notes:
    ------
    This function reads the output files for a given
    CMFGEN model and returns the blackbody emission
    and the rate of emitted ionising photons. The
    rate of emitted ionising photons are calculated
    by integrating the spectrum and the blackbody
    emission from the given model. The integration
    is done from the ionising limit to infinity.
    The blackbody emission is calculated from the
    effective temperature and radius given in the
    VADAT file. If it is not specified the temperature
    must be given to this function. The spectrum is read from the
    obs_fin file. 
    

    Author:     Ylva Götberg
    Date:       27/10 - 2015   

    """


    
    # Some of my own packages

    cstes = constants()
    h_SI = cstes['h_SI']
    c_SI = cstes['c_SI']
    kB_SI = cstes['kB_SI']
    RSun_SI = cstes['RSun_SI']
    E_H_lim_SI = cstes['E_H_lim_SI']
    lam_H_lim_SI = cstes['lam_H_lim_SI']
    E_HeI_lim_SI = cstes['E_HeI_lim_SI']
    lam_HeI_lim_SI = cstes['lam_HeI_lim_SI']
    E_HeII_lim_SI = cstes['E_HeII_lim_SI']
    lam_HeII_lim_SI = cstes['lam_HeII_lim_SI']
    
    
    # Get the effective temperature -- this is from MESA - only for the Blackbody!!! (for comparison)
    # Read the VADAT file, get the [TEFF] and radius
    with open(model_loc+'/VADAT','r') as f:
        tmp = f.readlines()
        for j in range(len(tmp)):
            if len(tmp[j]) > 1:
                tmp2 = tmp[j].split()
                if tmp2[1] == '[TEFF]':
                    Teff = float(tmp2[0])*1e4      # Effective temperature in K
                if tmp2[1] == '[RSTAR]':
                    radius = float(tmp2[0])*1e8    # Radius in meters
    
    surf_area = 4.0*np.pi*(radius**2.0)
   
    """
    # Read the MOD_SUM file, get radius and effective temperature
    with open(model_loc+'/MOD_SUM','r') as f:
        MOD_SUM = f.readlines()
        for j in range(len(MOD_SUM)):
            if 'Tau=6.667E-01' in MOD_SUM[j]:
                tmp = MOD_SUM[j].split()[3]
                Teff = float(tmp[8:])               # Effective temperature in K
                tmp = MOD_SUM[j].split()[2]
                radius = float(tmp[6:])*RSun_SI     # Effective radius in meter
    """    
    
    # Get the spectrum
    obs_loc = model_loc+'/obs/obs_fin'
    data = GetCMFGENflux(model_loc,obs_loc)
    F_lambda = data[0]
    lambda_SI = data[1]
    
    # Calculate the blackbody B_lambda from the given variables
    
    # B_lambda is in units [W m^-1]
    B_lambda = (np.pi*(radius**2.0))*(4.0*np.pi)*(2.0*h_SI*(c_SI**2.0)/(lambda_SI**5.0))/(np.exp(h_SI*c_SI/(lambda_SI*kB_SI*Teff)) - 1.0)

    
    # Calculate the Q0, Q1 and Q2
    # # # # First the blackbody
    # Q0 (hydrogen ionising photons / second)
    E_min_Q0 = E_H_lim_SI
    E_max_Q0 = np.infty
    Q0_bb = integrate_BB_star(Teff,radius,E_min_Q0,E_max_Q0)

    # Q1 (HeI ionising photons / second)
    E_min_Q1 = E_HeI_lim_SI
    E_max_Q1 = np.infty
    Q1_bb = integrate_BB_star(Teff,radius,E_min_Q1,E_max_Q1)

    # Q2 (HeII ionising photons / second)
    E_min_Q2 = E_HeII_lim_SI
    E_max_Q2 = np.infty
    Q2_bb = integrate_BB_star(Teff,radius,E_min_Q2,E_max_Q2)

    
    # # # # Then the spectrum
    
    # Q0 
    ind_Q0 = lambda_SI < lam_H_lim_SI
    l_h_1 = lambda_SI[ind_Q0][:-1]
    l_h_2 = lambda_SI[ind_Q0][1:]
    Q0_s = (1.0/(h_SI*c_SI))*np.sum(lambda_SI[ind_Q0][:-1]*F_lambda[ind_Q0][:-1]*np.abs(l_h_2 - l_h_1))

    # Q1
    ind_Q1 = lambda_SI < lam_HeI_lim_SI
    l_heI_1 = lambda_SI[ind_Q1][:-1]
    l_heI_2 = lambda_SI[ind_Q1][1:]
    Q1_s = (1.0/(h_SI*c_SI))*np.sum(lambda_SI[ind_Q1][:-1]*F_lambda[ind_Q1][:-1]*np.abs(l_heI_2 - l_heI_1))

    # Q2
    ind_Q2 = lambda_SI < lam_HeII_lim_SI
    l_heII_1 = lambda_SI[ind_Q2][:-1]
    l_heII_2 = lambda_SI[ind_Q2][1:]
    Q2_s = (1.0/(h_SI*c_SI))*np.sum(lambda_SI[ind_Q2][:-1]*F_lambda[ind_Q2][:-1]*np.abs(l_heII_2 - l_heII_1))

    
    # Return the Blackbody emission and the rate of emitted ionising photons
    return B_lambda, Q0_bb, Q1_bb, Q2_bb, Q0_s, Q1_s, Q2_s
    