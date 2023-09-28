# Extrapolation function
import numpy as np
import copy

def extrapolate(x, y, xmod, ymod):
    """

    Parameters:
    -----------
    x : array
        The x-values of the model data
    y : array
        The y-values of the model data
    xmod : array
        The x-values of the model to extrapolate from
    ymod : array
        The y-values of the model to extrapolate from
    
    Returns:
    --------
    y_new : array
        The extrapolated y-values
    
    Notes:
    ------
    This function extrapolates the model data to the limits of the model.
    The extrapolation is done by fitting a straight line to the two points
    closest to the limit and then extrapolating from that line.

    """
    
    # Sort the model data to extrapolate from 
    x = np.array(x)
    y = np.array(y)
    xmod = np.array(xmod)
    ymod = np.array(ymod)
    ind_sort = np.argsort(xmod)
    xmod = xmod[ind_sort]
    ymod = ymod[ind_sort]
    
    # Copy the array to send back an extrapolated version
    y_new = copy.copy(y)
        
    # Find the lower limits
    ind_min = x<xmod[0]
    if np.sum(ind_min):
        k = (ymod[1]-ymod[0])/(xmod[1]-xmod[0])
        m = ymod[0]-k*xmod[0]

        # Update the array
        y_new[ind_min] = k*x[ind_min] + m
    
    # Find the upper limits
    ind_max = x>xmod[-1]
    if np.sum(ind_max):
        k = (ymod[-2]-ymod[-1])/(xmod[-2]-xmod[-1])
        m = ymod[-1]-k*xmod[-1]

        # Update the array
        y_new[ind_max] = k*x[ind_max] + m    
    
    return y_new
