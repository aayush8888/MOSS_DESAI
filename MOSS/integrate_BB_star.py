# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import quad

# Own package
from constants import constants
"""
Created on Wed Jan 21 14:08:31 2015

@author: ylva
"""

""" = = = = = integrate_BB_star.py  = = = = = = = = = = """

""" This function integrates a blackbody (BB) spectrum
    of a star with the given temperature and radius and 
    between the given energies (=wavelengths).
    
    Returns the number of photons emitted per second.
    
    Author:     Ylva Götberg
    Date:       21/1 - 2015                             """
""" = = = = = = = = = = = = = = = = = = = = = = = = = = """

# Define this as a function
def integrate_BB_star(T,R,E_min,E_max):

    # Constants
    cstes = constants()
    h = cstes['h_SI']        # Planck's constant [m² kg s⁻¹]
    kB = cstes['kB_SI']      # Boltzmann's constant [m² kg s⁻² K⁻¹]
    c = cstes['c_SI']        # Speed of light [m s⁻¹]
    
    # Constant for integration
    #     From full sphere on sky (solid angle): 4*pi
    #     From surface of star: pi*R^2  (just a disk!!!)
    #     From energy per photon: 1/(h*c)
    #     From Blambda: 2*h*c^2
    #     From variable change: kB^3*T^3/(h^3*c^3)
    K = 8.0*(np.pi**2.0)*(R**2.0)*(kB**3.0)*(T**3.0)/((h**3.0)*(c**2.0))
    
    # To integrate, I use the variable change 
    #   u = h*c/(lambda*kB*T)
    #  
    # This variable change corresponds also to 
    #   u = E/(kB*T)
    #
    # Function to integrate
    # def u_func(u_var):
    #     return (u_var**2.0)/(np.exp(u_var) - 1.0)
    
    # Integrate the function
    
    E_tmp = h*c/(1e-9)
    if E_max > E_tmp: E_max = E_tmp
    
    u_min = E_min/(kB*T)
    u_max = E_max/(kB*T)
    #I_u = quad(u_func,u_min,u_max)
    #Q = K*I_u[0]
    nd = 1000
    uu = np.linspace(u_min,u_max,nd)
    ff = K*(uu**2.0)/(np.exp(uu)-1.0)
    QQ = ff[:-1]*(np.abs(uu[:-1]-uu[1:]))
    Q = np.sum(QQ)
    
    #print 'Check that integrate_BB_star.py does what you want!!!'
    # I have tested this now, it seems fine, the blackbody emits similar to the spectrum, but 
    # a bit more, as expected. Also I changed from using the integration function to the simpler
    # summation.
        
    return Q
