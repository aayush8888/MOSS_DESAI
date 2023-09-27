# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 16:16:55 2014

@author: ylva
"""

""" = = = = = GetColumnMESAhist.py  = = = = = = = = = = """

""" This function reads a specified history.data file 
    from the MESA output and gives the desired columns
    back.
    
    Author:     Ylva Götberg
    Date:       14/12 - 2014                             """
""" = = = = = = = = = = = = = = = = = = = = = = = = = = """

import numpy as np

# This is a function
def GetColumnMESAhist(filename_history,col):
    
    # Read the header of the history.data file
    try:
        header_entries = open(filename_history, "r").readlines()[5]
        header_entries = header_entries.split() 
        
        # Read the history.data file
        history = np.loadtxt(filename_history, skiprows=6)
        
        # Create the list which will be filled with data
        data_cols = list()
        
        # Iterate through the desired columns    
        for c in range(0,len(col)):
            ind = header_entries.index(col[c])
            
            if len(history) == len(header_entries):
                data_cols.append(history[ind])
            else:
                data_cols.append(history[:,ind])
    except:

        data_cols = list()        
        
    return data_cols