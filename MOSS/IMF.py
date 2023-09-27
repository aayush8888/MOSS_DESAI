# -*- coding: utf-8 -*-

""" = = = = = IMF.py  = = = = = = = = = = = = = = = = = """

""" This script holds IMFs as functions where inserted 
    total mass of a population gives a set of randomly 
    drawn masses for the stars following the IMF.
    
    Author:     Ylva Götberg
    Date:       22/6 - 2015                             """
""" = = = = = = = = = = = = = = = = = = = = = = = = = = """

# Import some packages
import numpy as np

# # # The Salpeter IMF (Salpeter 1955)
# 
# N(m) ~ m^-2.35
#
def Salpeter_IMF(U_rdm_array, mmin, mmax):
    
    # alpha for the Salpeter IMF is 2.35
    alpha = 2.35

    # This translates to n
    n = -alpha
        
    # The corresponding mass distribution is
    masses = (U_rdm_array*(mmax**(n+1) - mmin**(n+1)) + mmin**(n+1))**(1/(n+1))
    
    return masses


# # # The Kroupa IMF (Kroupa 2001)
#
# N(m) ~ m^-2.3      , m > 0.5 MSun
#        m^-1.3      , 0.08 < m < 0.5 MSun
#        m^-0.3      , m < 0.08
#
def Kroupa_IMF(U_rdm_array, mmin, mmax):   
    
    # Powers in the power laws
    a_alpha = -0.3 
    a_beta = -1.3
    a_gamma = -2.3
    
    # Where the power laws break
    m_1 = 0.08
    m_2 = 0.5  
    if m_1 < mmin:
        m_1 = mmin
    if m_2 < mmin:
        m_2 = mmin
    
    # Get the constants K_alpha, K_beta, K_gamma
    # 1) Boundary conditions (K_beta, K_gamma in terms of K_alpha)
    # 2) Normalising function
    j_alpha = (m_1**(a_alpha+1) - mmin**(a_alpha+1))/(a_alpha+1)
    j_beta = (m_2**(a_beta+1) - m_1**(a_beta+1))/(a_beta+1)
    j_gamma = (mmax**(a_gamma+1) - m_2**(a_gamma+1))/(a_gamma+1)
    c_beta = m_1**(a_alpha-a_beta)
    c_gamma = m_2**(a_beta-a_gamma)
    K_alpha = 1/(j_alpha + c_beta*j_beta + c_beta*c_gamma*j_gamma)
    K_beta = K_alpha * (m_1**(a_alpha-a_beta))
    K_gamma = K_beta * (m_2**(a_beta-a_gamma))
    
    # Fractions of the three power laws
    x = (K_alpha*(m_1**(a_alpha+1) - mmin**(a_alpha+1))/(a_alpha+1))
    y = (K_beta*(m_2**(a_beta+1) - m_1**(a_beta+1))/(a_beta+1))
    z = (K_gamma*(mmax**(a_gamma+1) - m_2**(a_gamma+1))/(a_gamma+1))
    
    # Choose which part of the broken power law to pick from
    D = np.random.rand(U_rdm_array.size)
    masses = np.zeros(U_rdm_array.size)
    ind_x = (D < x)
    ind_y = ((D > x)*(D < x+y))
    ind_z = (D > x+y)
    masses[ind_x] = ((U_rdm_array[ind_x]*(m_1**(a_alpha+1)-mmin**(a_alpha+1))+
                     mmin**(a_alpha+1))**(1/(a_alpha+1)))
    masses[ind_y] = ((U_rdm_array[ind_y]*(m_2**(a_beta+1)-m_1**(a_beta+1))+
                     m_1**(a_beta+1))**(1/(a_beta+1)))
    masses[ind_z] = ((U_rdm_array[ind_z]*(mmax**(a_gamma+1)-m_2**(a_gamma+1))+
                     m_2**(a_gamma+1))**(1/(a_gamma+1)))
                     
    return masses



# # # Alternative Kroupa IMF 
#
# N(m) ~ m^-a_gamma      , m > 0.5 MSun     Suggestions: -1.9 (Schneider+18), -2.7 (clusters??)
#        m^-1.3          , 0.08 < m < 0.5 MSun
#        m^-0.3          , m < 0.08
#
def Alt_Kroupa_IMF(U_rdm_array, mmin, mmax, a_gamma):   
    
    # Powers in the power laws
    a_alpha = -0.3 
    a_beta = -1.3
    #a_gamma = -2.3
    
    # Where the power laws break
    m_1 = 0.08
    m_2 = 0.5  
    if m_1 < mmin:
        m_1 = mmin
    if m_2 < mmin:
        m_2 = mmin
    
    # Get the constants K_alpha, K_beta, K_gamma
    # 1) Boundary conditions (K_beta, K_gamma in terms of K_alpha)
    # 2) Normalising function
    j_alpha = (m_1**(a_alpha+1) - mmin**(a_alpha+1))/(a_alpha+1)
    j_beta = (m_2**(a_beta+1) - m_1**(a_beta+1))/(a_beta+1)
    j_gamma = (mmax**(a_gamma+1) - m_2**(a_gamma+1))/(a_gamma+1)
    c_beta = m_1**(a_alpha-a_beta)
    c_gamma = m_2**(a_beta-a_gamma)
    K_alpha = 1/(j_alpha + c_beta*j_beta + c_beta*c_gamma*j_gamma)
    K_beta = K_alpha * (m_1**(a_alpha-a_beta))
    K_gamma = K_beta * (m_2**(a_beta-a_gamma))
    
    # Fractions of the three power laws
    x = (K_alpha*(m_1**(a_alpha+1) - mmin**(a_alpha+1))/(a_alpha+1))
    y = (K_beta*(m_2**(a_beta+1) - m_1**(a_beta+1))/(a_beta+1))
    z = (K_gamma*(mmax**(a_gamma+1) - m_2**(a_gamma+1))/(a_gamma+1))
    
    # Choose which part of the broken power law to pick from
    D = np.random.rand(U_rdm_array.size)
    masses = np.zeros(U_rdm_array.size)
    ind_x = (D < x)
    ind_y = ((D > x)*(D < x+y))
    ind_z = (D > x+y)
    masses[ind_x] = ((U_rdm_array[ind_x]*(m_1**(a_alpha+1)-mmin**(a_alpha+1))+
                     mmin**(a_alpha+1))**(1/(a_alpha+1)))
    masses[ind_y] = ((U_rdm_array[ind_y]*(m_2**(a_beta+1)-m_1**(a_beta+1))+
                     m_1**(a_beta+1))**(1/(a_beta+1)))
    masses[ind_z] = ((U_rdm_array[ind_z]*(mmax**(a_gamma+1)-m_2**(a_gamma+1))+
                     m_2**(a_gamma+1))**(1/(a_gamma+1)))
                     
    return masses
