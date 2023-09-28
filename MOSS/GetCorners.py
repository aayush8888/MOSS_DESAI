#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import astropy.units as u
# # # # More simple: GetCorners
# Another function that gets the corners of the interpolation grid in T and log g
def GetCorners(Tstar,log_g,T_Kurucz,log_g_Kurucz):
    """

    Parameters:
    -----------
    Tstar : array
        The temperatures of the stars
    log_g : array
        The surface gravities of the stars
    T_Kurucz : array
        The temperatures of the Kurucz models
    log_g_Kurucz : array
        The surface gravities of the Kurucz models
    
    Returns:
    --------
    ind_c1 : array
        The indices of the corners of the Kurucz grid
    ind_c2 : array
        The indices of the corners of the Kurucz grid
    ind_c3 : array
        The indices of the corners of the Kurucz grid
    ind_c4 : array
        The indices of the corners of the Kurucz grid
    f1 : array
        The interpolation factors for the Kurucz grid
    f2 : array
        The interpolation factors for the Kurucz grid
    ind_Kurucz_possible : array
        The indices of the stars that are covered by the Kurucz grid
    
    Notes:
    ------
    This function finds the indices of the corners of the Kurucz grid
    for the given stars. It also finds the interpolation factors for
    the Kurucz grid. The interpolation is done in log T and log g.
    The interpolation is done in the following way:
    f1 = (T1-T2)/(T3-T2)
    f2 = (g1-g2)/(g3-g2)
    where T1 and T2 are the temperatures of the star and the two
    Kurucz models closest to the star, and T3 is the temperature of
    the Kurucz model closest to the star but with a higher temperature.
    The same goes for the surface gravity.
    The indices of the corners are:
    c1 = (T1,g1)
    c2 = (T1,g2)
    c3 = (T3,g2)
    c4 = (T3,g1)
    where T1 and T3 are the temperatures of the Kurucz models closest
    to the star but with a lower and higher temperature, respectively.
    The same goes for the surface gravity.

    """

    RSun_SI = u.R_sun.to(u.m)          # Solar radius in meters

    # Find the closest temperatures and surface gravities for the different models
    [TK,TT] = np.meshgrid(T_Kurucz,Tstar)
    [gK,gg] = np.meshgrid(log_g_Kurucz,log_g)

    #ind_Kurucz_possible = [True]*len(Tstar)
    nbr_Kurucz = len(T_Kurucz)

    # Get the minimum and maximum temperature values for the Kurucz model grid
    ind = (TK-TT) < 0
    T_min = np.max(TK*ind,axis=1)
    T_min = np.array([T_min]).T*np.ones([1,nbr_Kurucz])
    # I need a workaround for the models that are hotter than the grid
    ind = (TK-TT) > 0
    tmp = TK*ind
    tmp[tmp == 0] = 1e6
    T_max = np.min(tmp,axis=1)
    T_max = np.array([T_max]).T*np.ones([1,nbr_Kurucz])

    # Get also the minimum and maximum surface graavity valiies for the Kurucz model grid
    ind = (gK-gg) < 0
    logg_min = np.max(gK*ind,axis=1)
    logg_min = np.array([logg_min]).T*np.ones([1,nbr_Kurucz])
    ind = (gK-gg) > 0
    tmp = gK*ind
    tmp[tmp == 0.] = 1e6
    logg_max = np.min(tmp,axis=1)
    logg_max = np.array([logg_max]).T*np.ones([1,nbr_Kurucz])
    del TT, gg

    # Find the corners for each star
    ind_Tmin = TK == T_min
    ind_Tmax = TK == T_max
    ind_loggmin = gK == logg_min
    ind_loggmax = gK == logg_max
    del T_min, T_max, logg_min, logg_max
    ind_c1 = ind_Tmin*ind_loggmin
    ind_c2 = ind_Tmin*ind_loggmax
    ind_c3 = ind_Tmax*ind_loggmax
    ind_c4 = ind_Tmax*ind_loggmin
    del ind_Tmin, ind_Tmax, ind_loggmin, ind_loggmax



    # These are the stars that are covered by the Kurucz grid:
    ind_Kurucz_possible = np.sum(ind_c1+ind_c2+ind_c3+ind_c4, axis=1) == 4
    ind_c1 = ind_c1[ind_Kurucz_possible,:]
    ind_c2 = ind_c2[ind_Kurucz_possible,:]
    ind_c3 = ind_c3[ind_Kurucz_possible,:]
    ind_c4 = ind_c4[ind_Kurucz_possible,:]

    # INTERPOLATION OF SPECTRA
    T1 = np.log10(Tstar)[ind_Kurucz_possible]
    T2 = np.log10(TK[ind_Kurucz_possible,:][ind_c1])
    T3 = np.log10(TK[ind_Kurucz_possible,:][ind_c4])
    f1 = (T1-T2)/(T3-T2)
    g1 = log_g[ind_Kurucz_possible]
    g2 = gK[ind_Kurucz_possible,:][ind_c1]
    g3 = gK[ind_Kurucz_possible,:][ind_c2]
    f2 = (g1-g2)/(g3-g2)
    del TK, gK, T1, T2, T3, g1, g2, g3

    return ind_c1, ind_c2, ind_c3, ind_c4, f1, f2, ind_Kurucz_possible
