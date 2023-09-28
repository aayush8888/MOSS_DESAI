# -*- coding: utf-8 -*-
import numpy as np


def constants():

    """
    = = = = = constants.py  = = = = = = = = = = = = = = 

    Parameters:
    -----------
    None

    Returns:
    --------
    const : dict
        Dictionary with constants in SI and cgs units.
    
    Notes:
    ------
    The constants are taken from the NIST website:
    http://physics.nist.gov/cuu/Constants/index.html

    
    Author:     Ylva Götberg
    Date:       8/7 - 2015                              
    """ 
    
    # Boltzmann's constant
    kB_SI = 1.3806488e-23
    kB_cgs = 1.3806488e-16

    # Electron volt in Joule
    eV_SI = 1.60217657e-19
    erg_SI = 1e-7    # 1 erg in Joules
    J_cgs = 1e7      # 1 Joule in ergs

    # Planck's constant
    h_SI = 6.62606957e-34
    h_cgs = 6.62606957e-27

    # Stefan Boltzmann constant    
    sigma_SB_SI = 5.670373e-8
    sigma_SB_cgs = 5.6704e-5   
    
    # Speed of light
    c_SI = 2.99792458e8         # m/s
    c_cgs = 2.99792458e10    # cm/s

    # Ionising limits
    E_H_lim_eV = 13.5984
    E_H_lim_SI = E_H_lim_eV*eV_SI
    lam_H_lim_SI = h_SI*c_SI/E_H_lim_SI
    
    E_HeI_lim_eV = 24.5874
    E_HeI_lim_SI = E_HeI_lim_eV*eV_SI
    lam_HeI_lim_SI = h_SI*c_SI/E_HeI_lim_SI
    
    E_HeII_lim_eV = 54.417760
    E_HeII_lim_SI = E_HeII_lim_eV*eV_SI
    lam_HeII_lim_SI = h_SI*c_SI/E_HeII_lim_SI
    
    # Solar luminosity in SI
    LSun_SI = 3.846e26     # Watt
    LSun_cgs = LSun_SI*1e7   # erg/s
    
    # Solar mass in SI
    MSun_SI = 1.989e30     # kg
    
    # Solar radius 
    RSun_SI = 6.957e8    # meters
    RSun_AU = 0.0046491    # AU
    
    # Parsec in meters
    pc_SI = 3.08567758149137e16
    
    # AU in meters
    AU_SI = 1.496e+11

    # Gravitational constant
    G_SI = 6.67408e-11		# m^3 kg^-1 s^-2
    G_cgs = 6.67408e-8		# cm^3 g^-1 s^-2
    G_Eorb = 4*(np.pi**2.0)     # AU^3 MSun^-1 yr^-2
    G_orb_days = G_Eorb/(365.25**2.0)     # AU^3 MSun^-1 days^-2
    
    # Time units
    yr_SI = 365.25*24.*60.*60.     # Years in seconds
    day_SI = 24.*60.*60.
    
    const = dict(h_SI=h_SI,h_cgs=h_cgs,sigma_SB_SI=sigma_SB_SI,sigma_SB_cgs=sigma_SB_cgs,
                 c_SI=c_SI, c_cgs=c_cgs, kB_SI=kB_SI, kB_cgs=kB_cgs, eV_SI=eV_SI, E_H_lim_eV=E_H_lim_eV, 
                 E_HeI_lim_eV=E_HeI_lim_eV, E_HeII_lim_eV=E_HeII_lim_eV, 
                 E_H_lim_SI=E_H_lim_SI, E_HeI_lim_SI=E_HeI_lim_SI, 
                 E_HeII_lim_SI=E_HeII_lim_SI, lam_H_lim_SI=lam_H_lim_SI, 
                 lam_HeI_lim_SI=lam_HeI_lim_SI, lam_HeII_lim_SI=lam_HeII_lim_SI,
                 RSun_SI=RSun_SI, RSun_AU=RSun_AU, pc_SI=pc_SI, AU_SI=AU_SI, G_SI=G_SI, 
                 G_cgs=G_cgs, G_Eorb=G_Eorb, G_orb_days=G_orb_days, 
                 LSun_SI=LSun_SI, LSun_cgs=LSun_cgs, MSun_SI=MSun_SI, yr_SI=yr_SI, day_SI=day_SI,
                 erg_SI=erg_SI)
    
    return const
