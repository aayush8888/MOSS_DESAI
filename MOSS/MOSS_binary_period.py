import numpy as np
import matplotlib.pyplot as plt
from extrapolate import extrapolate
import copy

class COMPUTE_BINARIES_period(object):
    """
    Helper class with functions to assign period to the binaries. The assignment is based on the mass of the primary and the period distribution is chosen by the user.
    """
    

    def __init__(self, period_choice, nbr_bin, primary, q, nbr_stars, m1, m2, birthday, birthday_m2, single, m, m_grid, iii,log_name, plots_dir, save_figs, fsize, Mlim_Sana, RSun_AU, G, R_ZAMS_grid, P_max, kappa_P):
            """
            Initialize the class
            """
            self.log_name = log_name
            self.plots_dir = plots_dir
            self.fsize = fsize
            self.RSun_AU = RSun_AU
            self.G = G
            self.R_ZAMS_grid = R_ZAMS_grid
            self.m_grid = m_grid
            self.P_max = P_max
            self.Mlim_Sana = Mlim_Sana
            self.kappa_P = kappa_P
            self.update_params(nbr_bin,iii,primary, q, nbr_stars, m1, m2, birthday, birthday_m2, single, m)


            
            if save_figs:
                self.plot_func = self.plotter
            else:
                self.plot_func = self.plotter_dummy
        
            if period_choice == 'Opik_Sana':
                self.period_func = self.period_choice_Opik_Sana
            elif period_choice == 'log_power':
                self.period_func = self.period_choice_log_power
            elif period_choice == 'log_flat':
                self.period_func = self.period_choice_log_flat


    def update_params(self, nbr_bin,iii,primary, q, nbr_stars, m1, m2, birthday, birthday_m2, single, m):

        self.nbr_bin = nbr_bin
        self.iii = iii
        self.primary = primary
        self.q = q
        self.nbr_stars = nbr_stars
        self.m1 = m1
        self.m2 = m2
        self.birthday = birthday
        self.birthday_m2 = birthday_m2
        self.single = single
        self.m = m

        self.R_ZAMS = np.interp(self.m1,self.m_grid,self.R_ZAMS_grid)
        self.R_ZAMS = extrapolate(self.m1,self.R_ZAMS,self.m_grid,self.R_ZAMS_grid)

        # Translate the above radii to periods at which the 
        # star would fill its Roche Lobe
        self.q_inv = 1.0/self.q    # This m1/m2
        self.rL = 0.49*(self.q_inv**(2.0/3.0))/(0.69*(self.q_inv**(2.0/3.0)) + np.log(1.0+(self.q_inv**(1.0/3.0))))

        # Period needed to fill Roche Lobe at ZAMS
        self.a_tmp = (self.R_ZAMS/self.rL)*self.RSun_AU  # separation in AU
        self.P_ZAMS = (4.0*(np.pi**2.0)*(self.a_tmp**3.0)/(self.G*(self.m1+self.m2)))**0.5    # Period in days

        # Now to the period limits
        self.P_min = copy.copy(self.P_ZAMS)

    def period_choice_Opik_Sana(self):
            """
            The combination of Opik (1924) and Sana et al. (2012)
            """

        # Period distribution. This is mass dependent. 
            self.P = np.zeros(self.nbr_bin)

            # # # # Massive stars
            # Sana distribution (Sana+ 12):   dN/d(log P) = k*(log P)^-0.55

            # When to apply the Sana period distribution? At O-star masses
            self.ind_Sana = self.m1 >= self.Mlim_Sana
            self.nbr_Sana = np.sum(self.ind_Sana)

            # Draw from the distribution
            self.uu = np.random.random(self.nbr_Sana)
            self.kappa_P_Sana = -0.55    # this is from Sana+12
            self.smin = np.log10(self.P_min[self.ind_Sana])
            self.smin[self.smin < 0.15] = 0.15   # this is also from Sana+12 and lower doesn't work because of maths
            self.smax = np.log10(self.P_max)
            self.s = (self.smin**(self.kappa_P_Sana+1.) + self.uu*(self.smax**(self.kappa_P_Sana+1.) - self.smin**(self.kappa_P_Sana+1.)))**(1./(self.kappa_P_Sana+1.))
            self.P_Sana = 10**self.s
            self.P[self.ind_Sana] = self.P_Sana


            # # # # Lower masses
            # Opik's law -- I choose all other masses in this period distribution
            self.ind_Opik = self.ind_Sana == 0
            self.nbr_Opik = np.sum(self.ind_Opik)

            # Opik's law is flat in log P space
            uu = np.random.random(self.nbr_Opik)
            P_Opik = 10**((np.log10(self.P_max)-np.log10(self.P_min[self.ind_Opik]))*uu + np.log10(self.P_min[self.ind_Opik]))
            self.P[self.ind_Opik] = P_Opik

            # Tell the log
            
            self.write_log('Used the combination of Opik and Sana+12 for the period distribution\n')
            self.write_log('The minimum period for the Sana+12 distribution is '+str(10**0.15)+'days, while the Opik distribution goes to minimum possible period.\n')
            self.write_log('Ignoring widening via wind mass loss.\n')
            
            

         
    def period_choice_log_power(self):
        """
        Power law:  dN/d(log P) = k*(log P)^kappa_P
        """
        self.uu = np.random.random(self.nbr_bin)
        self.smin = np.log10(self.P_min)
        self.smin[self.smin < 0.15] = 0.15   # This is so that the function below works (~1.4 days)
        self.smax = np.log10(self.P_max)
        self.s = (self.smin**(self.kappa_P+1.) + self.uu*(self.smax**(self.kappa_P+1.) - self.smin**(self.kappa_P+1.)))**(1./(self.kappa_P+1.))
        self.P = 10**self.s

        self.write_log('Used the a power-law distribution that scales as ~ (log P)^'+str(self.kappa_P)+'\n')
        self.write_log('Ignoring widening via wind mass loss.\n')
         
    def period_choice_log_flat(self):
        """
        Flat in log P (meaning favors short periods still, Opik 24)
        """
        self.P = 10**(np.log10(self.P_min)+(np.log10(self.P_max)-np.log10(self.P_min))*np.random.random(self.nbr_bin))
        self.write_log('Used the a power-law distribution that scales as ~ (log P)^0\n')
        self.write_log('Ignoring widening via wind mass loss.\n')
         
    def plotter(self,Opik_Sana=False,log_power=False,log_flat=False):
        """
        Function that plots the periods for the corresponding distribution.
        """
        clr_l = np.array([ 212, 172, 13 ])/255.
        clr_h = np.array([211, 84, 0])/255.
        fig, ax = plt.subplots(1,1,figsize=(6,4.5))
        if Opik_Sana:
            ax.hist(np.log10(self.P[self.m1>self.Mlim_Sana]),100,color=clr_h,edgecolor='none')
            ax.hist(np.log10(self.P[self.m1<self.Mlim_Sana][0:np.sum(self.m1>15)]),100,color=clr_l,edgecolor='none',alpha=0.7)
        if log_power or log_flat:
             ax.hist(np.log10(self.P),100,color='b',edgecolor='none')
                 
        ax.set_xlabel('Period [days]')
        ax.set_ylabel('Number of stars')
        xtick = [0,1,2,3]
        ax.set_xticks(xtick)
        ax.set_xticklabels([1, 10, 100, 1000])
        for i in range(len(xtick)):
            ax.get_xaxis().majorTicks[i].set_pad(7)
        ax.tick_params('both', length=8, width=1.5, which='major')
        ax.tick_params('both', length=4, width=1.0, which='minor')
        fig.savefig(self.plots_dir+'/P_fewstars.png',format='png',bbox_inches='tight',pad_inches=0.1)
        plt.close(fig)

        self.write_log('Saved a plot in '+self.plots_dir+'\n\n')
        self.plot_func = self.plotter_dummy


    def plotter_dummy(self):
        pass


    def write_log(self,message):
        # Tell the log
        fid_log = open(self.log_name,'a')
        fid_log.write(message)
        fid_log.close()

    def __call__(self):
        self.period_func()
        self.plot_func()
        return self.s,self.P


    

# Tell the log
    
    
    

        # # # # # #  Period, P # # # # # # 
        # 
        # Period is picked randomly from distributions 

        # Tell the log
        
        
        

        # Period limits:
        # Pmin is P_ZAMS 
        # Pmax = 10^3.5 or 10^3.7 days
        # Starting from radius limits

        # Interpolate using the numpy
        # R_ZAMS = np.interp(m1,m_grid,R_ZAMS_grid)
        # R_ZAMS = extrapolate(m1,R_ZAMS,m_grid,R_ZAMS_grid)

        # # Translate the above radii to periods at which the 
        # # star would fill its Roche Lobe
        # q_inv = 1.0/q    # This m1/m2
        # rL = 0.49*(q_inv**(2.0/3.0))/(0.69*(q_inv**(2.0/3.0)) + np.log(1.0+(q_inv**(1.0/3.0))))

        # # Period needed to fill Roche Lobe at ZAMS
        # a_tmp = (R_ZAMS/rL)*RSun_AU  # separation in AU
        # P_ZAMS = (4.0*(np.pi**2.0)*(a_tmp**3.0)/(G*(m1+m2)))**0.5    # Period in days

        # # Now to the period limits
        # P_min = copy.copy(P_ZAMS)        # in Sana+12 this is 10^0.15
        # # The maximum period is set at the beginning. 


        # # The combination of Opik (1924) and Sana et al. (2012) 
        # if P_choice == 'Opik_Sana':

        #     # Period distribution. This is mass dependent. 
        #     P = np.zeros(nbr_bin)

        #     # # # # Massive stars
        #     # Sana distribution (Sana+ 12):   dN/d(log P) = k*(log P)^-0.55

        #     # When to apply the Sana period distribution? At O-star masses
        #     ind_Sana = m1 >= Mlim_Sana
        #     nbr_Sana = np.sum(ind_Sana)

        #     # Draw from the distribution
        #     uu = np.random.random(nbr_Sana)
        #     kappa_P_Sana = -0.55    # this is from Sana+12
        #     smin = np.log10(P_min[ind_Sana])
        #     smin[smin < 0.15] = 0.15   # this is also from Sana+12 and lower doesn't work because of maths
        #     smax = np.log10(P_max)
        #     s = (smin**(kappa_P_Sana+1.) + uu*(smax**(kappa_P_Sana+1.) - smin**(kappa_P_Sana+1.)))**(1./(kappa_P_Sana+1.))
        #     P_Sana = 10**s
        #     P[ind_Sana] = P_Sana


        #     # # # # Lower masses
        #     # Opik's law -- I choose all other masses in this period distribution
        #     ind_Opik = ind_Sana == 0
        #     nbr_Opik = np.sum(ind_Opik)

        #     # Opik's law is flat in log P space
        #     uu = np.random.random(nbr_Opik)
        #     P_Opik = 10**((np.log10(P_max)-np.log10(P_min[ind_Opik]))*uu + np.log10(P_min[ind_Opik]))
        #     P[ind_Opik] = P_Opik

        #     if iii == 0:
        #         if save_figs:
        #             # # # # Plot the Period # # # # # # # #
        #             clr_l = np.array([ 212, 172, 13 ])/255.
        #             clr_h = np.array([211, 84, 0])/255.
        #             fig, ax = plt.subplots(1,1,figsize=(6,4.5))
        #             ax.hist(np.log10(P[m1>Mlim_Sana]),100,color=clr_h,edgecolor='none')
        #             ax.hist(np.log10(P[m1<Mlim_Sana][0:np.sum(m1>15)]),100,color=clr_l,edgecolor='none',alpha=0.7)
        #             ax.set_xlabel('Period [days]')
        #             ax.set_ylabel('Number of stars')
        #             xtick = [0,1,2,3]
        #             ax.set_xticks(xtick)
        #             ax.set_xticklabels([1, 10, 100, 1000])
        #             for i in range(len(xtick)):
        #                 ax.get_xaxis().majorTicks[i].set_pad(7)
        #             ax.tick_params('both', length=8, width=1.5, which='major')
        #             ax.tick_params('both', length=4, width=1.0, which='minor')
        #             fig.savefig(plots_dir+'/P_fewstars.png',format='png',bbox_inches='tight',pad_inches=0.1)
        #             plt.close(fig)
        #             # # # # # # # # # # # # # # # # #


        #     # Tell the log
            
        #     utils_object.write_log('Used the combination of Opik and Sana+12 for the period distribution\n')
        #     utils_object.write_log('The minimum period for the Sana+12 distribution is '+str(10**0.15)+'days, while the Opik distribution goes to minimum possible period.\n')
        #     utils_object.write_log('Ignoring widening via wind mass loss.\n')
        #     if save_figs:
        #         utils_object.write_log('Saved a plot in '+plots_dir+'\n\n')
            


        # # Power law:  dN/d(log P) = k*(log P)^kappa_P
        # elif P_choice == 'log_power':

        #     # Period distribution.
        #     uu = np.random.random(nbr_bin)
        #     smin = np.log10(P_min)
        #     smin[smin < 0.15] = 0.15   # This is so that the function below works (~1.4 days)
        #     smax = np.log10(P_max)
        #     s = (smin**(kappa_P+1.) + uu*(smax**(kappa_P+1.) - smin**(kappa_P+1.)))**(1./(kappa_P+1.))
        #     P = 10**s       


        #     if iii == 0:
        #         if save_figs:
        #             # # # # Plot the Period # # # # # # # #
        #             fig, ax = plt.subplots(1,1,figsize=(6,4.5))
        #             ax.hist(np.log10(P),100,color='b',edgecolor='none')
        #             ax.set_xlabel('Period [days]')
        #             ax.set_ylabel('Number of stars')
        #             xtick = [0,1,2,3]
        #             ax.set_xticks(xtick)
        #             ax.set_xticklabels([1, 10, 100, 1000])
        #             for i in range(len(xtick)):
        #                 ax.get_xaxis().majorTicks[i].set_pad(7)
        #             ax.tick_params('both', length=8, width=1.5, which='major')
        #             ax.tick_params('both', length=4, width=1.0, which='minor')
        #             fig.savefig(plots_dir+'/P.png',format='png',bbox_inches='tight',pad_inches=0.1)
        #             plt.close(fig)
        #             # # # # # # # # # # # # # # # # #


        #     # Tell the log
            
        #     utils_object.write_log('Used the a power-law distribution that scales as ~ (log P)^'+str(kappa_P)+'\n')
        #     utils_object.write_log('Ignoring widening via wind mass loss.\n')
        #     if save_figs:
        #         utils_object.write_log('Saved a plot in '+plots_dir+'\n\n')
            


        # # Flat in log P (meaning favors short periods still, Opik 24)
        # elif P_choice == 'log_flat':

        #     # Period distribution. This is mass dependent. 
        #     P = 10**(np.log10(P_min)+(np.log10(P_max)-np.log10(P_min))*np.random.random(nbr_bin))

        #     if iii == 0:
        #         if save_figs:
        #             # # # # Plot the Period # # # # # # # #
        #             fig, ax = plt.subplots(1,1,figsize=(6,4.5))
        #             ax.hist(np.log10(P),100,color='b',edgecolor='none')
        #             ax.set_xlabel('Period [days]')
        #             ax.set_ylabel('Number of stars')
        #             xtick = [0,1,2,3]
        #             ax.set_xticks(xtick)
        #             ax.set_xticklabels([1, 10, 100, 1000])
        #             for i in range(len(xtick)):
        #                 ax.get_xaxis().majorTicks[i].set_pad(7)
        #             ax.tick_params('both', length=8, width=1.5, which='major')
        #             ax.tick_params('both', length=4, width=1.0, which='minor')
        #             fig.savefig(plots_dir+'/P.png',format='png',bbox_inches='tight',pad_inches=0.1)
        #             plt.close(fig)
        #             # # # # # # # # # # # # # # # # #


        #     # Tell the log
            
        #     utils_object.write_log('Used the a power-law distribution that scales as ~ (log P)^0\n')
        #     utils_object.write_log('Ignoring widening via wind mass loss.\n')
        #     if save_figs:
        #         utils_object.write_log('Saved a plot in '+plots_dir+'\n\n')
            
    
    


