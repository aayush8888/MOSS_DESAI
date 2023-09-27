# -*- coding: utf-8 -*-
"""
"""

""" = = = = = locate_ZAMS.py  = = = = = = = = = = = = = """

""" This function returns the index in data arrays which
    corresponds to the ZAMS. 
    
    Author:     Ylva Götberg
    Date:       21/4 - 2015                             """
""" = = = = = = = = = = = = = = = = = = = = = = = = = = """

# Define as a function
def locate_ZAMS(logL, logLnuc):

    # Import some packages
    import numpy as np
    
    # Create Lnuc/L
    Lnuc_div_L = (10**logLnuc)/(10**logL)

    # Create a list of indices as long as the luminosity arrays
    len_L = len(logL)
    indices = np.arange(0,len_L,1)

    # Get the first index at which Lnuc_div_L is close to 1
    tmp = (Lnuc_div_L > 0.9)*(Lnuc_div_L < 1.1)
    tmp2 = indices[tmp]
    ind_close_to_1 = tmp2[0]

    # Take the coming 100 indices after the first in ind_close_to_1 
    # and check when the Lnuc_div_L has stabilised -- then the 
    # ZAMS has begun
    clip = 500
    first_clip = np.arange(ind_close_to_1,ind_close_to_1+clip,1)
    if (ind_close_to_1+clip > len_L):
        first_clip = np.arange(ind_close_to_1,len_L,1)

    # Now check when the Lnuc_div_L starts to vary less than 5%
    # for more than 10 outputs in a row
    var_lim = 0.01
    s = 50
    index = 0
    n = 0
    while ((index == 0) and (n+s < clip)):
        fn = first_clip[n]
        ln = first_clip[n+s]
        check = np.sum(np.abs((Lnuc_div_L[fn:ln])-1) > var_lim)
        #print np.abs((Lnuc_div_L[fn]))
        if (check == 0):
            index = fn
            #print "Found the index!"
        n = n + 1
        if (n+s > clip):
            print("Change in the locate_ZAMS.py script!")
             

    return index
