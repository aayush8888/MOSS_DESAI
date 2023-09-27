# -*- coding: utf-8 -*-

""" = = = = = IBF.py  = = = = = = = = = = = = = = = = = """

""" This script allows for testing various initial B-field functions (IBFs), 
as well as defining the ratio of magnetic to non-magnetic stars in the population. 
While the concept of the IBF is analogous to the IMF, there is an additional variable, 
the above mentioned ratio or incidence rate. 
The *currently observed* incidence rate is established to be roughly 10%
in the Galaxy, based on observations of ~1,000 stars. It is unknown in other metallicities.
The *currently observed* distribution of magnetic field strengths is roughly lognormal (Shultz+2019).
Thus the main questions are: 
What is the IBF that allows for reproducing the currently observed range of field strength, and
what is the initial B_frac that allows for the currently determined incidence rate.
    
    Authors:    Ylva Götberg and Zsolt Keszthelyi
    Year:       2022                           """
""" = = = = = = = = = = = = = = = = = = = = = = = = = = """

# These are the free parameters in the main code: 
#'IBF_choice', 'Bmin' , 'Bmax' , 'Bmean' , 'Bstdev' , 'Bfrac', 'Bk', 'Bmu' 
# Not all are needed for all IBFs, check a bit which ones you need. 

import numpy as np
from scipy.stats import *

# ---------------------------
# Uniform = flat distribution
# ---------------------------
# If the IBF is flat, then all values are uniformly distributed with equal weight. 
# The same number of stars will have, say, 3 and 30 kG fields. 
def IBF_flat(nbr_B, Bmin, Bmax): 
    B = Bmin + (Bmax-Bmin)*np.random.random(nbr_B)
    # Locate the types of magnetic fields not allowed and draw again for them
    ind_redraw = (B <= 0.) + (B < Bmin) + (B > Bmax) > 0.
    while np.sum(ind_redraw):
        nbr_redraw = np.sum(ind_redraw)
        B_tmp = Bmin + (Bmax-Bmin)*np.random.random(nbr_redraw)
        B[ind_redraw] = B_tmp
        ind_redraw = (B <= 0.) + (B < Bmin) + (B > Bmax) > 0.
    return B

# -----------------
# Gaussian (normal) 
# -----------------
# YG: Zsolt, I think this one now in fact is a Poisson distribution - is that what we want? Perhaps we just want to use the Poisson instead?
def IBF_gaussian(nbr_B, Bmin, Bmax, Bmean, Bstdev):
    # Assign normally distributed B-field strength for magnetic stars
    B = np.random.normal(Bmean, Bstdev, nbr_B)
    # Locate the types of magnetic fields not allowed and draw again for them
    ind_redraw = (B <= 0.) + (B < Bmin) + (B > Bmax) > 0.
    while np.sum(ind_redraw):
        nbr_redraw = np.sum(ind_redraw)
        B_tmp = np.random.normal(Bmean, Bstdev, nbr_redraw)
        B[ind_redraw] = B_tmp
        ind_redraw = (B <= 0.) + (B < Bmin) + (B > Bmax) > 0.
    return B

# Zsolt: 
# %%% --- NEW --- %%% 
# check a few interesting distributions. 
# detailed work planned by Veronique and her student. 
# %%% --- === --- %%% 

# ---------
# lognormal 
# ---------
# Zsolt: My bet is on this one. I think this is the best match for the observed Galactic stars, according to 
# Shultz+2019. The main question: is the currently observed distribution the same as the ZAMS one???
#
def IBF_log10normal(nbr_B, Bmin, Bmax, Bmean, Bstdev):
    # Assign lognormally distributed B-field strength for magnetic stars
    #B = np.random.lognormal(Bmean, Bstdev, nbr_B)  # YG: I think this is natural logarithm - don't we want to have 10-logarithm? 
    # Make sure we don't have negative B-field strengths or extrapolate outside grid
    #B[B<Bmin] = Bmin
    #B[B>Bmax] = Bmax
    
    # YG: Is this instead what we are looking for? 
    B = 10**np.random.normal(Bmean, Bstdev, nbr_B)
    # Locate the types of magnetic fields not allowed and draw again for them
    ind_redraw = (B <= 0.) + (B < Bmin) + (B > Bmax) > 0.
    while np.sum(ind_redraw):
        nbr_redraw = np.sum(ind_redraw)
        B_tmp = 10**np.random.normal(Bmean, Bstdev, nbr_redraw)
        B[ind_redraw] = B_tmp
        ind_redraw = (B <= 0.) + (B < Bmin) + (B > Bmax) > 0.    
    return B

# --------
# Poission
# --------
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html?highlight=poisson#scipy.stats.poisson
def IBF_poisson(nbr_B, Bmin, Bmax, Bk, Bmu):
    # Assign poisson distributed B-field strength for magnetic stars
    B = np.random.possion(Bk, Bmu, nbr_B)
    # Make sure we don't have negative B-field strengths or extrapolate outside grid
    #B[B<Bmin] = Bmin
    #B[B>Bmax] = Bmax
    # Locate the types of magnetic fields not allowed and draw again for them
    ind_redraw = (B <= 0.) + (B < Bmin) + (B > Bmax) > 0.
    while np.sum(ind_redraw):
        nbr_redraw = np.sum(ind_redraw)
        B_tmp = np.random.poisson(Bk, Bmu, nbr_redraw)
        B[ind_redraw] = B_tmp
        ind_redraw = (B <= 0.) + (B < Bmin) + (B > Bmax) > 0.    
    return B

# add other distribution functions if required. 
# ---------
# Maxwell 
# ---------
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.maxwell.html
# ---------
# chi
# ---------
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi.html#scipy.stats.chi


