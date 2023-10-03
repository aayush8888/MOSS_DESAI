import numpy as np
import matplotlib.pyplot as plt

class COMPUTE_BINARIES(object):
    """ Helper Class for all functions corresponding to binary star computations"""
    def __init__(self,m,nbr_stars,iii,log_name,save_figs,plots_dir,mmin,mmax,fbin_constant,fbin_choice):
        self.update_params(m,nbr_stars,iii,mmin,mmax)
        self.log_name = log_name
        self.plots_dir = plots_dir
        self.mmin = mmin
        self.mmax = mmax
        self.fbin_constant = fbin_constant

        if save_figs:
            self.plot_func = self.plotter
        else:
            self.plot_func = self.plotter_dummy

        if fbin_choice == 'DK':
            self.fbin_func = self.fbin_choice_DK
        elif fbin_choice == 'vH13':
            self.fbin_func = self.fbin_choice_vH13
        elif fbin_choice == 'moe':
            self.fbin_func = self.fbin_choice_moe
        elif fbin_choice == 'constant':
            self.fbin_func = self.fbin_choice_constant




    def update_params(self,m,nbr_stars,iii,mmin,mmax):

        self.m = m
        self.nbr_stars = nbr_stars
        self.iii = iii
        self.mmin = mmin
        self.mmax = mmax

    def fbin_choice_DK(self):
        """
        Function using the binary fraction from Duchene & Kraus (2013)
        """
        self.message = 'Using the binary fraction from Duchene and Kraus (2013)\n'
        self.write_log()

        # The binary fraction is mass dependent
        self.primary = np.zeros(self.nbr_stars)

        # Very low mass stars are not included in this analysis but Duchene & Kraus do mention
        # something about the companion fraction so just in case I put here that Aller (2007) 
        # estimated the multiplicity fraction for m <~ 0.1 MSun to be 22^{+6}_{-4} %. 

        # Low-mass stars data come from nearly complete samples as described in Duchene & Kraus (2013)
        # from Delfosse+ 04, Dieterich+ 12 and Reid & Gizis (97a)
        self.ind_low_mass = (self.m < 0.5)*(self.m >= 0.1)
        self.nbr_low_mass = np.sum(self.ind_low_mass)
        if self.nbr_low_mass>0:
            self.primary[self.ind_low_mass] = np.random.random(self.nbr_low_mass) <= 0.26

        # WHAT ABOUT 0.5 < m < 0.7 MSun ????? 

        # For the solar-mass stars I pick from Duchene & Kraus (2013) data from Raghavan+ 2010
        # 0.7 < m < 1.0 MSun
        self.ind_solar_mass1 = (self.m < 1.0)*(self.m >= 0.7)
        self.nbr_solar_mass1 = np.sum(self.ind_solar_mass1)
        if self.nbr_solar_mass1>0:
            self.primary[self.ind_solar_mass1] = np.random.random(self.nbr_solar_mass1) <= 0.41

        # 1.0 < m < 1.3 MSun
        self.ind_solar_mass2 = (self.m < 1.3)*(self.m >= 1.0)
        self.nbr_solar_mass2 = np.sum(self.ind_solar_mass2)
        if self.nbr_solar_mass2>0:
            self.primary[self.ind_solar_mass2] = np.random.random(self.nbr_solar_mass2) <= 0.5

        # WHAT ABOUT 1.3 < m < 1.5 MSun ??????

        # Intermediate mass stars 
        # Duchene & Kraus are not very precise here, but say the multiplicity fraction is > 0.5
        # 1.5 < m < 5.0 MSun
        self.ind_intermediate_mass = (self.m < 5.0)*(self.m >= 1.5)
        self.nbr_intermediate_mass = np.sum(self.ind_intermediate_mass)
        if self.nbr_intermediate_mass>0:
            self.primary[self.ind_intermediate_mass] = np.random.random(self.nbr_intermediate_mass) <= 0.5

        # High mass stars
        # For high-mass stars I will just bluntly use 0.7. It is very optimistic in Duchene & Kraus.
        self.ind_high_mass = self.m >= 5.0
        self.nbr_high_mass = np.sum(self.ind_high_mass)
        if self.nbr_high_mass>0:
            self.primary[self.ind_high_mass] = np.random.random(self.nbr_high_mass) <= 0.69

        # Total number of binary stars
        self.nbr_bin = np.sum(self.primary)

        # Make primary a boolean array
        self.primary = self.primary == 1

        self.plot_func(self,DK=True)

        return self.primary,self.nbr_bin

    def fbin_choice_vH13(self):
        """
        Function using the binary fraction from van der van Haaften et al. (2013)
        """

        # Tell the log
        self.message = 'Using the binary fraction from van Haaften et al. (2013)\n'
        self.write_log()

        # Create a random array 
        temp_random = np.random.random(self.nbr_stars)

        # Get the binary fraction for each mass
        self.fbin = 0.5 + 0.25*np.log10(self.m)
        self.fbin[self.fbin>1.0] = 1.0

        # Determine which stars are in binary
        self.primary = temp_random <= self.fbin

        # How many binaries is that?
        self.nbr_bin = np.sum(self.primary)

        self.plot_func(self,vH13=True)

        return self.primary,self.nbr_bin


    def fbin_choice_moe(self):
        """
        Function using the binary fraction from Moe & DiStefano (2017)
        """
        # Tell the log
        self.message = 'Using a function fitted to Fig. 42 of Moe & DiStefano (2017) for the binary fraction \n'
        self.write_log()

        # Get the binary fraction for each mass
        # steps: 
        # (1) read the Fig 42 of Moe & DiStefano (2017), 
        #filename_moe = '/data001/ygoetberg/taurus/python_scripts/ionisation_paper/moe_distefano_17.txt'
        #data = np.loadtxt(filename_moe)
        #M_tmp = data[:,0]
        #fbin_tmp = data[:,1]
        # This file contains the binary fractions for interacting binaries that Moe & DiStefano (2017) found (their Fig. 42), for 0.2 < log10 P < 3.7 and q > 0.1
        # I have used plot digitizer to get these values.
        self.M_tmp = np.array([28.208070697850278,12.194394238503687,7.013958043119244,3.5179000994706486,1.0033899071551038])
        self.fbin_tmp = np.array([1.039580885942532,0.7743880876109841,0.621660551424101,0.3699230117699013,0.1392481270452426])
        # (2) fit line, 
        self.coeff = np.polyfit(np.log10(self.M_tmp),self.fbin_tmp,1)
        # (3) use line as function
        self.fbin = np.polyval(self.coeff,np.log10(self.m))
        self.fbin[self.fbin>1.] = 1.

        # Create a random array
        self.temp_random = np.random.random(self.nbr_stars)

        # Determine which stars are in binary
        self.primary = self.temp_random <= self.fbin

        # How many binaries is that? 
        self.nbr_bin = np.sum(self.primary)

        self.plot_func(self,moe=True)

        return self.primary,self.nbr_bin

    
    def fbin_choice_constant(self):
        
        self.message = 'Assuming a mass-independent binary fraction of fbin = '+str(self.fbin_constant)+'\n'
        self.write_log()

        self.temp_random = np.random.random(self.nbr_stars)

        # Determine which stars are in binary
        self.primary = self.temp_random <= self.fbin_constant

        # How many binaries is that? 
        self.nbr_bin = np.sum(self.primary)

        self.plot_func(self,constant=True)

        return self.primary,self.nbr_bin

    
    def plotter(self,DK=False, vH13=False, moe=False, constant=False):
        """
        Function to plot the binary fraction.
        """

        fig, ax = plt.subplots(1,1,figsize=(6,4.5))
        clr_l = np.array([118, 68, 138])/255.
        clr_f = np.array([175, 122, 197])/255.
    
        if DK:
            mm = np.logspace(0,2.5,101)
            fbin_plot = np.zeros(len(mm)-1)
            mid_m = np.zeros(len(mm)-1)
            for i in range(len(mm)-1):
                bin_bin = (self.m[self.primary] >= mm[i])*(self.m[self.primary] < mm[i+1])
                sin_bin = (self.m >= mm[i])*(self.m < mm[i+1])
                if np.sum(sin_bin) >0:
                    fbin_plot[i] = float(np.sum(bin_bin))/float(np.sum(sin_bin))
                mid_m[i] = mm[i]+(mm[i+1]-mm[i])/2.0
                ytick = [0,0.2,0.4,0.6,0.8,1.0]
                ax.set_yticks(ytick)
                ax.set_xlim([self.mmin,self.mmax])
        
        if vH13:
            clr_l = np.array([118, 68, 138])/255.
            clr_f = np.array([175, 122, 197])/255.
            mm = np.logspace(0,2.5,100)
            fbin_plot = 0.5+0.25*np.log10(mm)
            fbin_plot[fbin_plot>1.0] = 1.0
            ax.set_xlim([1,100])
            ax.set_ylim([0,1])

        if moe:
            mm = np.logspace(0,2.5,100)
            fbin_plot = self.coeff[0]*np.log10(mm) + self.coeff[1]
            fbin_plot[fbin_plot>1.0] = 1.0
            ax.set_xlim([1,100])
            ax.set_ylim([0,1.05])
            ax.tick_params(direction="in", which='both')

        if constant:
            self.medges = np.logspace(np.log10(self.mmin), np.log10(self.mmax),21)
            self.mmid = 10**(np.log10(self.medges[:-1])+ (np.log10(self.medges[1:])-np.log10(self.medges[:-1]))/2.)
            self.fbin_plot = np.zeros(len(self.medges)-1)
            for k in range(len(self.medges)-1):
                self.ind_mbin = (self.m>=self.medges[k])*(self.m<self.medges[k+1])
                self.fbin_plot[k] = np.sum(self.primary*self.ind_mbin)/np.float_(np.sum(self.ind_mbin))
            ax.set_xlim([1,100])
            ax.set_ylim([0,1.05])
            ax.tick_params(direction="in", which='both')



        ax.semilogx(mid_m,fbin_plot,'-',color=clr_l,lw=3)
        if moe or vH13:
            ax.fill_between(mm,np.zeros(len(fbin_plot)),fbin_plot,color=clr_f)

        ax.set_xlabel(r'Mass [$M_{\\odot}$]')
        ax.set_ylabel(r'$f_{{bin}}$')
        xtick = [1,10,100]
        ax.set_xticks(xtick)
        ax.set_xticklabels([1,10,100])
        for i in range(len(xtick)):
            ax.get_xaxis().majorTicks[i].set_pad(7)
        ax.tick_params('both', length=8, width=1.5, which='major')
        ax.tick_params('both', length=4, width=1.0, which='minor')
        fig.savefig(self.plots_dir+'/fbin.png',format='png',bbox_inches='tight',pad_inches=0.1)
        plt.close(fig)

        self.plot_func = self.plotter_dummy


    def write_log(self):
        # Tell the log
        fid_log = open(self.log_name,'a')
        fid_log.write(self.message)
        fid_log.close()


    def plotter_dummy(self):
        pass


    def __call__(self):
        return self.fbin_func()

    # if compute_binaries:

    #     # # # # # #  Binary fraction, f_bin # # # # # # 
    #     # The binary fraction is drawn from distributions

    #     # Tell the log
    #     
    #     utils_object.write_log('BINARY FRACTION: Going to pick which stars are in binaries \n')
    #     

    #     # Duchene & Kraus (2013)
    #     if fbin_choice == 'DK':

    #         # Tell the log
    #         
    #         utils_object.write_log('Using the binary fraction from Duchene and Kraus (2013)\n')
    #         

    #         # The binary fraction is mass dependent
    #         primary = np.zeros(nbr_stars)

    #         # Very low mass stars are not included in this analysis but Duchene & Kraus do mention
    #         # something about the companion fraction so just in case I put here that Aller (2007) 
    #         # estimated the multiplicity fraction for m <~ 0.1 MSun to be 22^{+6}_{-4} %. 

    #         # Low-mass stars data come from nearly complete samples as described in Duchene & Kraus (2013)
    #         # from Delfosse+ 04, Dieterich+ 12 and Reid & Gizis (97a)
    #         ind_low_mass = (m < 0.5)*(m >= 0.1)
    #         nbr_low_mass = np.sum(ind_low_mass)
    #         if nbr_low_mass>0:
    #             primary[ind_low_mass] = np.random.random(nbr_low_mass) <= 0.26

    #         # WHAT ABOUT 0.5 < m < 0.7 MSun ????? 

    #         # For the solar-mass stars I pick from Duchene & Kraus (2013) data from Raghavan+ 2010
    #         # 0.7 < m < 1.0 MSun
    #         ind_solar_mass1 = (m < 1.0)*(m >= 0.7)
    #         nbr_solar_mass1 = np.sum(ind_solar_mass1)
    #         if nbr_solar_mass1>0:
    #             primary[ind_solar_mass1] = np.random.random(nbr_solar_mass1) <= 0.41

    #         # 1.0 < m < 1.3 MSun
    #         ind_solar_mass2 = (m < 1.3)*(m >= 1.0)
    #         nbr_solar_mass2 = np.sum(ind_solar_mass2)
    #         if nbr_solar_mass2>0:
    #             primary[ind_solar_mass2] = np.random.random(nbr_solar_mass2) <= 0.5

    #         # WHAT ABOUT 1.3 < m < 1.5 MSun ??????

    #         # Intermediate mass stars 
    #         # Duchene & Kraus are not very precise here, but say the multiplicity fraction is > 0.5
    #         # 1.5 < m < 5.0 MSun
    #         ind_intermediate_mass = (m < 5.0)*(m >= 1.5)
    #         nbr_intermediate_mass = np.sum(ind_intermediate_mass)
    #         if nbr_intermediate_mass>0:
    #             primary[ind_intermediate_mass] = np.random.random(nbr_intermediate_mass) <= 0.5

    #         # High mass stars
    #         # For high-mass stars I will just bluntly use 0.7. It is very optimistic in Duchene & Kraus.
    #         ind_high_mass = m >= 5.0
    #         nbr_high_mass = np.sum(ind_high_mass)
    #         if nbr_high_mass>0:
    #             primary[ind_high_mass] = np.random.random(nbr_high_mass) <= 0.69

    #         # Total number of binary stars
    #         nbr_bin = np.sum(primary)

    #         # Make primary a boolean array
    #         primary = primary == 1


    #         if iii == 0:
    #             if save_figs:
    #                 # # # # Plot the fbin # # # # # # # #
    #                 mm = np.logspace(0,2.5,101)
    #                 fbin_plot = np.zeros(len(mm)-1)
    #                 mid_m = np.zeros(len(mm)-1)
    #                 for i in range(len(mm)-1):
    #                     bin_bin = (m[primary] >= mm[i])*(m[primary] < mm[i+1])
    #                     sin_bin = (m >= mm[i])*(m < mm[i+1])
    #                     if np.sum(sin_bin) >0:
    #                         fbin_plot[i] = float(np.sum(bin_bin))/float(np.sum(sin_bin))
    #                     mid_m[i] = mm[i]+(mm[i+1]-mm[i])/2.0

    #                 fig, ax = plt.subplots(1,1,figsize=(6,4.5))
    #                 ax.semilogx(mid_m,fbin_plot,'b-',lw=2)
    #                 ax.set_xlabel('Mass [$M_{\\odot}$]')
    #                 ax.set_ylabel('$f_{\\mathrm{bin}}$')
    #                 xtick = [1,10,100]
    #                 ax.set_xticks(xtick)
    #                 ax.set_xticklabels([1,10,100])
    #                 for i in range(len(xtick)):
    #                     ax.get_xaxis().majorTicks[i].set_pad(7)
    #                 ax.tick_params('both', length=8, width=1.5, which='major')
    #                 ax.tick_params('both', length=4, width=1.0, which='minor')
    #                 ytick = [0,0.2,0.4,0.6,0.8,1.0]
    #                 ax.set_yticks(ytick)
    #                 ax.set_xlim([mmin,mmax])
    #                 fig.savefig(plots_dir+'/fbin.png',format='png',bbox_inches='tight',pad_inches=0.1)
    #                 plt.close(fig)
    #                 # # # # # # # # # # # # # # # # # # 



    #     # The analytical relation of van Haaften et al. (2013)
    #     elif fbin_choice == 'vH13':

    #         # Tell the log
    #         
    #         utils_object.write_log('Using the binary fraction from van Haaften et al. (2013)\n')
    #         

    #         # Create a random array 
    #         temp_random = np.random.random(nbr_stars)

    #         # Get the binary fraction for each mass
    #         fbin = 0.5 + 0.25*np.log10(m)
    #         fbin[fbin>1.0] = 1.0

    #         # Determine which stars are in binary
    #         primary = temp_random <= fbin

    #         # How many binaries is that?
    #         nbr_bin = np.sum(primary)

    #         if iii == 0:
    #             if save_figs:
    #                 # # # # Plot the fbin # # # # # # # #
    #                 clr_l = np.array([118, 68, 138])/255.
    #                 clr_f = np.array([175, 122, 197])/255.
    #                 mm = np.logspace(0,2.5,100)
    #                 fbin_plot = 0.5+0.25*np.log10(mm)
    #                 fbin_plot[fbin_plot>1.0] = 1.0
    #                 fig, ax = plt.subplots(1,1,figsize=(6,4.5))
    #                 ax.semilogx(mm,fbin_plot,'-',color=clr_l,lw=3)
    #                 ax.fill_between(mm,np.zeros(len(fbin_plot)),fbin_plot,color=clr_f)
    #                 ax.set_xlabel('Mass [$M_{\\odot}$]')
    #                 ax.set_ylabel('$f_{\\mathrm{bin}}$')
    #                 xtick = [1,10,100]
    #                 ax.set_xticks(xtick)
    #                 ax.set_xticklabels([1,10,100])
    #                 for i in range(len(xtick)):
    #                     ax.get_xaxis().majorTicks[i].set_pad(7)
    #                 ax.tick_params('both', length=8, width=1.5, which='major')
    #                 ax.tick_params('both', length=4, width=1.0, which='minor')
    #                 ax.set_xlim([1,100])
    #                 ax.set_ylim([0,1])
    #                 fig.savefig(plots_dir+'/fbin.png',format='png',bbox_inches='tight',pad_inches=0.1)
    #                 plt.close(fig)
    #                 # # # # # # # # # # # # # # # # # # 


    #     # This binary fraction is from Moe & DiStefano (2017), see their Fig. 42
    #     elif fbin_choice == 'moe':

    #         # Tell the log
    #         
    #         utils_object.write_log('Using a function fitted to Fig. 42 of Moe & DiStefano (2017) for the binary fraction \n')
    #         

    #         # Get the binary fraction for each mass
    #         # steps: 
    #         # (1) read the Fig 42 of Moe & DiStefano (2017), 
    #         #filename_moe = '/data001/ygoetberg/taurus/python_scripts/ionisation_paper/moe_distefano_17.txt'
    #         #data = np.loadtxt(filename_moe)
    #         #M_tmp = data[:,0]
    #         #fbin_tmp = data[:,1]
    #         # This file contains the binary fractions for interacting binaries that Moe & DiStefano (2017) found (their Fig. 42), for 0.2 < log10 P < 3.7 and q > 0.1
    #         # I have used plot digitizer to get these values.
    #         M_tmp = np.array([28.208070697850278,12.194394238503687,7.013958043119244,3.5179000994706486,1.0033899071551038])
    #         fbin_tmp = np.array([1.039580885942532,0.7743880876109841,0.621660551424101,0.3699230117699013,0.1392481270452426])
    #         # (2) fit line, 
    #         coeff = np.polyfit(np.log10(M_tmp),fbin_tmp,1)
    #         # (3) use line as function
    #         fbin = np.polyval(coeff,np.log10(m))
    #         fbin[fbin>1.] = 1.

    #         # Create a random array
    #         temp_random = np.random.random(nbr_stars)

    #         # Determine which stars are in binary
    #         primary = temp_random <= fbin

    #         # How many binaries is that? 
    #         nbr_bin = np.sum(primary)

    #         if iii == 0:
    #             if save_figs:
    #                 # # # # Plot the fbin # # # # # # # #
    #                 clr_l = np.array([118, 68, 138])/255.
    #                 clr_f = np.array([175, 122, 197])/255.
    #                 mm = np.logspace(0,2.5,100)
    #                 fbin_plot = coeff[0]*np.log10(mm) + coeff[1]
    #                 fbin_plot[fbin_plot>1.0] = 1.0
    #                 fig, ax = plt.subplots(1,1,figsize=(6,4.5))
    #                 ax.semilogx(mm,fbin_plot,'-',color=clr_l,lw=3)
    #                 ax.fill_between(mm,np.zeros(len(fbin_plot)),fbin_plot,color=clr_f)
    #                 ax.set_xlabel('Mass [$M_{\\odot}$]')
    #                 ax.set_ylabel('$f_{\\mathrm{bin}}$')
    #                 xtick = [1,10,100]
    #                 ax.set_xticks(xtick)
    #                 ax.set_xticklabels([1,10,100])
    #                 for i in range(len(xtick)):
    #                     ax.get_xaxis().majorTicks[i].set_pad(7)
    #                 ax.tick_params('both', length=8, width=1.5, which='major')
    #                 ax.tick_params('both', length=4, width=1.0, which='minor')
    #                 ax.set_xlim([1,100])
    #                 ax.set_ylim([0,1.05])
    #                 ax.tick_params(direction="in", which='both')
    #                 fig.savefig(plots_dir+'/fbin.png',format='png',bbox_inches='tight',pad_inches=0.1)
    #                 plt.close(fig)
    #                 # # # # # # # # # # # # # # # # # # 


    #     # Allow for a mass-independent binary fraction too
    #     elif fbin_choice == 'constant':

    #         # Tell the log
    #         
    #         utils_object.write_log('Assuming a mass-independent binary fraction of fbin = '+str(fbin_constant)+'\n')
    #             

    #         # Create a random array
    #         temp_random = np.random.random(nbr_stars)

    #         # Determine which stars are in binary
    #         primary = temp_random <= fbin_constant

    #         # How many binaries is that? 
    #         nbr_bin = np.sum(primary)

    #         if iii == 0:
    #             if save_figs:
    #                 # # # # Plot the fbin # # # # # # # #
    #                 clr_l = np.array([118, 68, 138])/255.
    #                 clr_f = np.array([175, 122, 197])/255.
    #                 fig, ax = plt.subplots(1,1,figsize=(6,4.5))
    #                 medges = np.logspace(np.log10(mmin), np.log10(mmax),21)
    #                 mmid = 10**(np.log10(medges[:-1])+ (np.log10(medges[1:])-np.log10(medges[:-1]))/2.)
    #                 fbin_plot = np.zeros(len(medges)-1)
    #                 for k in range(len(medges)-1):
    #                     ind_mbin = (m>=medges[k])*(m<medges[k+1])
    #                     fbin_plot[k] = np.sum(primary*ind_mbin)/np.float_(np.sum(ind_mbin))
    #                 ax.semilogx(mmid,fbin_plot,'-',color=clr_l)
    #                 ax.semilogx(mmid,fbin_plot,'.',color=clr_f)
    #                 ax.set_xlabel('Mass [$M_{\\odot}$]')
    #                 ax.set_ylabel('$f_{\\mathrm{bin}}$')
    #                 xtick = [1,10,100]
    #                 ax.set_xticks(xtick)
    #                 ax.set_xticklabels([1,10,100])
    #                 for i in range(len(xtick)):
    #                     ax.get_xaxis().majorTicks[i].set_pad(7)
    #                 ax.tick_params('both', length=8, width=1.5, which='major')
    #                 ax.tick_params('both', length=4, width=1.0, which='minor')
    #                 ax.set_xlim([1,100])
    #                 ax.set_ylim([0,1.05])
    #                 ax.tick_params(direction="in", which='both')
    #                 fig.savefig(plots_dir+'/fbin.png',format='png',bbox_inches='tight',pad_inches=0.1)
    #                 plt.close(fig)
    #                 # # # # # # # # # # # # # # # # # # 

    # Not necessary to do the above if there are only single stars
    # else: 
    #     nbr_bin = 0.
    #     primary = np.zeros(nbr_stars) != 0.