# -*- coding: utf-8 -*-
import numpy as np
def GetSpectra(loc_mod,file_type):
    """
    
    Parameters:
    -----------
    loc_mod : string
        The location of the model
    file_type : string
        The type of file to be read. Either 'obs' or 'OBSFLUX'

    Returns:
    --------
    lambda_mod : array
        The wavelengths in Angstrom
    Flambda_mod : array
        The fluxes in erg s^-1 cm^-2 A^-1
    Fnorm_mod : array
        The normalized fluxes
    Flambda_cont : array
        The continuum fluxes in erg s^-1 cm^-2 A^-1

    Notes:
    ------
    This function reads the output files for a given
    CMFGEN model and returns the wavelengths, frequencies
    and fluxes (Fnu and Flambda). The fluxes are assumed
    that all emission is seen (no 1 kpc assumption or
    similar). They are created for CMFGEN models and assuming that 
    you already created the obs and obs_cont directories with the spectra.
    """
    #from GetCMFGENflux import GetCMFGENflux
    from constants import constants
    
    # Get some constants
    cste = constants()
    c_SI = cste['c_SI']
    
    if file_type == 'obs':

        # Just assuming that the spectra are here
        obs_name = loc_mod+'/obs/obs_fin'
        obs_cont = loc_mod+'/obs_cont/obs_cont'


        # # # # # THE SPECTRUM # # # # # 
        # Open the obs_fin file
        fid = open(obs_name,'r')
        spec = fid.readlines()
        fid.close()

        # Storage for the frequency and flux
        freq = np.array([])
        Fnu = np.array([])
        type_content = ''
        # Loop over the content of the obs_fin file to get the frequencies and fluxes
        for i in range(len(spec)):
            if spec[i] != '\n':
                # Read frequencies in unit 10^15 Hz
                if 'Continuum Frequencies' in spec[i]:
                    type_content = 'freq'
                # Read flux at 1 kpc in unit Janskys (1e-23 erg s^-1 cm^-2 Hz^-1)
                elif 'Observed intensity' in spec[i]:
                    type_content = 'fnu'

                else:
                    if type_content == 'freq':
                        tmp = spec[i].split()
                        for k in range(len(tmp)):
                            if ('-' in tmp[k]) and (('E' not in tmp[k]) and ('D' not in tmp[k])):
                                tmp[k] = tmp[k].replace('-','E-')
                        freq = np.concatenate([freq,np.float_(tmp)])
                    elif type_content == 'fnu':
                        tmp = spec[i].split()
                        for k in range(len(tmp)):
                            if ('-' in tmp[k]) and (('E' not in tmp[k]) and ('D' not in tmp[k])):
                                tmp[k] = tmp[k].replace('-','E-')
                        Fnu = np.concatenate([Fnu,np.float_(tmp)])

        # Convert frequencies to wavelengths in Angstrom
        lambda_mod = (c_SI/(freq*1e15))*1e10   

        # Convert F_nu to F_lambda in cgs units
        Fnu_SI = Fnu*(1e-26)  # W m^-2 Hz^-1
        lambda_SI = lambda_mod/1e10   # meters
        Flambda_SI = (freq*1e15)*Fnu_SI/lambda_SI    # W m^-3 at 1 kpc
        Flambda_mod = Flambda_SI/1e7   # erg s^-1 cm^-2 A^-1 at 1 kpc



        # # # # # THE CONTINUUM # # # # # 
        # Open the obs_cont file
        fid = open(obs_cont,'r')
        cont = fid.readlines()
        fid.close()

        # Storage for the frequency and flux
        freq_cont = np.array([])
        Fnu_cont = np.array([])
        type_content = ''
        # Loop over the content of the obs_fin file to get the frequencies and fluxes
        for i in range(len(cont)):
            if spec[i] != '\n':
                # Read frequencies in unit 10^15 Hz
                if 'Continuum Frequencies' in cont[i]:
                    type_content = 'freq'
                # Read flux at 1 kpc in unit Janskys (1e-23 erg s^-1 cm^-2 Hz^-1)
                elif 'Observed intensity' in cont[i]:
                    type_content = 'fnu'

                else:
                    if type_content == 'freq':
                        tmp = cont[i].split()
                        for k in range(len(tmp)):
                            if ('-' in tmp[k]) and (('E' not in tmp[k]) and ('D' not in tmp[k])):
                                tmp[k] = tmp[k].replace('-','E-')
                        freq_cont = np.concatenate([freq_cont,np.float_(tmp)])
                    elif type_content == 'fnu':
                        tmp = cont[i].split()
                        for k in range(len(tmp)):
                            if ('-' in tmp[k]) and (('E' not in tmp[k]) and ('D' not in tmp[k])):
                                tmp[k] = tmp[k].replace('-','E-')
                        Fnu_cont = np.concatenate([Fnu_cont,np.float_(tmp)])

        # Convert frequencies to wavelengths in Angstrom
        lambda_cont = (c_SI/(freq_cont*1e15))*1e10   

        # Convert F_nu to F_lambda in cgs units
        Fnu_cont_SI = Fnu_cont*(1e-26)  # W m^-2 Hz^-1
        lambda_cont_SI = lambda_cont/1e10   # meters
        Flambda_cont_SI = (freq_cont*1e15)*Fnu_cont_SI/lambda_cont_SI    # W m^-3 at 1 kpc
        Flambda_cont = Flambda_cont_SI/1e7   # erg s^-1 cm^-2 A^-1 at 1 kpc    




        # # # # # THE NORMALIZED SPECTRUM # # # # # 

        # Need to first interpolate the continuum to the wavelength array of the SED
        Flambda_cont_interp = np.interp(lambda_mod,lambda_cont,Flambda_cont)
        Fnorm_mod = Flambda_mod/Flambda_cont_interp


        return lambda_mod, Flambda_mod, Fnorm_mod, Flambda_cont
    
    
    # This option is available for checking the where luminosity error comes from
    elif file_type ==  'OBSFLUX':
        
        # Location of the OBSFLUX file
        obsflux = loc_mod+'/OBSFLUX'


        # # # # # THE SPECTRUM # # # # # 
        # Open the obs_fin file
        fid = open(obsflux,'r')
        spec = fid.readlines()
        fid.close()

        # Storage for the frequency and flux
        freq = np.array([])
        Fnu = np.array([])
        type_content = ''
        # Loop over the content of the obs_fin file to get the frequencies and fluxes
        for i in range(len(spec)):
            if spec[i] != '\n':
                # Read frequencies in unit 10^15 Hz
                if 'Continuum Frequencies' in spec[i]:
                    type_content = 'freq'
                # Read flux at 1 kpc in unit Janskys (1e-23 erg s^-1 cm^-2 Hz^-1)
                elif 'Observed intensity' in spec[i]:
                    type_content = 'fnu'
                    
                elif 'Luminosity' in spec[i]:
                    type_content = 'luminosity'

                else:
                    if type_content == 'freq':
                        tmp = spec[i].split()
                        for k in range(len(tmp)):
                            if ('-' in tmp[k]) and (('E' not in tmp[k]) and ('D' not in tmp[k])):
                                tmp[k] = tmp[k].replace('-','E-')
                        freq = np.concatenate([freq,np.float_(tmp)])
                    elif type_content == 'fnu':
                        tmp = spec[i].split()
                        for k in range(len(tmp)):
                            if ('-' in tmp[k]) and (('E' not in tmp[k]) and ('D' not in tmp[k])):
                                tmp[k] = tmp[k].replace('-','E-')
                        Fnu = np.concatenate([Fnu,np.float_(tmp)])

        # Convert frequencies to wavelengths in Angstrom
        lambda_mod = (c_SI/(freq*1e15))*1e10   

        # Convert F_nu to F_lambda in cgs units
        Fnu_SI = Fnu*(1e-26)  # W m^-2 Hz^-1
        lambda_SI = lambda_mod/1e10   # meters
        Flambda_SI = (freq*1e15)*Fnu_SI/lambda_SI    # W m^-3 at 1 kpc
        Flambda_mod = Flambda_SI/1e7   # erg s^-1 cm^-2 A^-1 at 1 kpc  
        
        return lambda_mod, Flambda_mod
        
