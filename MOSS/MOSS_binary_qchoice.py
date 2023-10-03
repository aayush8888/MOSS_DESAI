import numpy as np
import copy
import matplotlib.pyplot as plt


class COMPUTE_BINARIES_qchoice(object):
    """
    Helper class with functions to assign masses to the secondaries
    """

    def __init__(self,q_choice,qmax,qmin,nbr_bin,iii,log_name,plots_dir,eta_q,m,primary,Target_M,save_figs,single,birthday,fsize):
        """
        Initialise the class
        """
        self.update_params(nbr_bin,iii,qmin,qmax,m,primary,birthday,single,Target_M,eta_q)
        self.log_name = log_name
        self.plots_dir = plots_dir
        self.fsize = fsize


        if save_figs:
            self.plot_func = self.plotter
        else:
            self.plot_func = self.plotter_dummy
    
        if q_choice == 'power_slope':
            self.q_func = self.q_choice_power_slope
        elif q_choice == 'flat':
            self.q_func = self.q_choice_flat
        elif q_choice == 'linear_slope':
            self.q_func = self.q_choice_linear_slope


    def update_params(self,nbr_bin,iii,qmin,qmax,m,primary,birthday,single,Target_M,eta_q):

        self.nbr_bin = nbr_bin
        self.iii = iii
        self.qmin = qmin
        self.qmax = qmax
        self.m = m
        self.primary = primary
        self.birthday = birthday
        self.single = single
        self.Target_M = Target_M
        self.eta_q = eta_q


    def write_log(self,message):
        # Tell the log
        fid_log = open(self.log_name,'a')
        fid_log.write(message)
        fid_log.close()

    def q_choice_flat(self):
        self.q = (self.qmax-self.qmin)*np.random.random(self.nbr_bin) + self.qmin
        self.write_log('Assigned masses to the secondaries following a flat distribution in mass ratio. \n')

    def q_choice_power_slope(self):

        # Random array between 0 and 1
        U = np.random.random(self.nbr_bin)

        # Solving the formel
        self.q = (self.qmin**(self.eta_q+1.) + U*(self.qmax**(self.eeta_q+1.) - self.qmin**(self.eta_q+1.)))**(1./(self.eta_q+1.))

        # Tell the log
        
        self.write_log('Assigned masses to the secondaries following a power-law distribution in mass ratio with exponent'+str(self.eta_q)+'. \n')


    def q_choice_linear_slope(self):
        # Ok, now I will use a 1D polynomial to draw mass ratios from
            # The distribution goes as dN/dq = eta_q*q + bb
            # Find the bb that normalizes the distribution
        bb = (1. - (self.eta_q/2.)*((self.qmax**2.) - (self.qmin**2.)))/(self.qmax-self.qmin)

        # This is the random numbers between 0 and 1, they will be used for when inverting the CDF to get the q
        U = np.random.random(self.nbr_bin)

        # Now solving the pq formel
        DD = -(self.eta_q/2.)*(self.qmin**2.) - bb*self.qmin - U
        EE = (2./self.eta_q)*DD
        q_minus = -(bb/self.eta_q) - np.sqrt((bb/self.eta_q)**2. - EE)
        q_plus = -(bb/self.eta_q) + np.sqrt((bb/self.eta_q)**2. - EE)
        self.q = np.zeros(self.nbr_bin)
        ind_qminus = (q_minus < self.qmax)*(q_minus > self.qmin)*(((q_plus < self.qmin)+(q_plus > self.qmax))>0.)
        ind_qplus = (q_plus < self.qmax)*(q_plus > self.qmin)*(((q_minus < self.qmin)+(q_minus > self.qmax))>0.)
        self.q[ind_qminus] = q_minus[ind_qminus]
        self.q[ind_qplus] = q_plus[ind_qplus]

        # Tell the log
        
        self.write_log('Assigned masses to the secondaries following a 1D polynomial distribution in mass ratio: dN/dq = '+str(self.eta_q)+'q + bb. \n')

    def plotter(self):
        """
        Function that plots the mass fraction for the corresponding distribution.
        """
        fig, ax = plt.subplots(1,1,figsize=(6,4.5))
        ax.hist(self.q,100)
        ax.set_xlabel('Mass ratio, $q$')
        ax.set_ylabel('Number stars')
        ax.set_yticks([])
        xtick = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ax.set_xticks(xtick)
        ax.set_xticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        for i in range(len(xtick)):
            ax.get_xaxis().majorTicks[i].set_pad(7)
        ax.tick_params('both', length=8, width=1.5, which='major')
        ax.tick_params('both', length=4, width=1.0, which='minor')
        ax.tick_params(direction="in", which='both')
        fig.savefig(self.plots_dir+'/q.png',format='png',bbox_inches='tight',pad_inches=0.1)
        plt.close(fig)
        # # # # # # # # # # # # # # # # # # 


        # # # Check the IMF again! # # #
        fig, ax1 = plt.subplots(1,1,figsize=(8,6))
        ax1.hist(np.log10(np.concatenate([self.m,self.m2])),100,log=True,label='IMF including M2')
        ax1.hist(np.log10(self.m),100,log=True,color='r',alpha=0.5,label='IMF excluding M2')
        ax1.set_xlabel('$\\log_{10} (M/M_{\\odot})$')
        ax1.set_ylabel('Number of stars')
        ax1.tick_params(direction="in", which='both')
        ax1.legend(loc=0,fontsize=0.8*self.fsize)
        fig.savefig(self.plots_dir+'/IMF_update_after_q.png',format='png',bbox_inches='tight',pad_inches=0.1)
        plt.close(fig)
        self.plot_func = self.plotter_dummy


    def plotter_dummy(self):
        pass


    def __call__(self):
        self.q_func()
        m2 = self.q*self.m[self.primary]

        # Now the total mass is too high -> scale down again to the total mass we should have.
        Mtot1 = np.sum(self.m)
        Mtot2 = np.sum(m2)
        Mtot = Mtot1 + Mtot2
        n = 0
        n2 = 0
        while Mtot > self.Target_M:
            Mtot1 = Mtot1-self.m[n]
            if self.primary[n]: 
                Mtot2 = Mtot2 - m2[n2]
                n2 = n2+1
            n = n+1
            Mtot = Mtot1+Mtot2
        self.m = self.m[n:]
        self.primary = self.primary[n:]
        self.single = self.single[n:]
        self.birthday = self.birthday[n:]
        # Total number of binary stars
        self.nbr_bin = np.sum(self.primary)
        # Total number of stars (counting binaries as 1 star)
        self.nbr_stars = len(self.m)

        # Update initial masses of primaries (m1) and secondaries (m2), initial mass ratio (q)
        self.m1 = self.m[self.primary]
        self.m2 = m2[n2:]
        self.q = self.q[n2:]
        # Set the birthday of the secondary to the same as the primary
        self.birthday_m2 = copy.copy(self.birthday[self.primary])

        self.plot_func()

        return self.primary,self.q,self.nbr_stars,self.nbr_bin,self.m1,self.m2,self.birthday,self.birthday_m2,self.single,self.m
    















            
        
        # # # # # #  Mass ratio, q # # # # # # 
        #
        # The mass ratio is drawn from distributions

        # Tell the log
        
        
        

        # # The flat mass ratio distribution
        # if q_choice == 'flat':

        #     # Mass ratio is a tricky business and so far flat between 0.1 and 1 is pretty standard.
        #     q = (qmax-qmin)*np.random.random(nbr_bin) + qmin

        #     # Tell the log
            
        #     utils_object.write_log('Assigned masses to the secondaries following a flat distribution in mass ratio. \n')
            


        # # This is the power-law slope:    dN/dq = k*q^eta_q
        # elif q_choice == 'power_slope':

        #     # Random array between 0 and 1
        #     U = np.random.random(nbr_bin)

        #     # Solving the formel
        #     q = (qmin**(eta_q+1.) + U*(qmax**(eta_q+1.) - qmin**(eta_q+1.)))**(1./(eta_q+1.))

        #     # Tell the log
            
        #     utils_object.write_log('Assigned masses to the secondaries following a power-law distribution in mass ratio with exponent'+str(eta_q)+'. \n')
            


        # # This is for 1D polynomial:    dN/dq = eta_q*q + bb 
        # elif q_choice == 'linear_slope':

        #     # Ok, now I will use a 1D polynomial to draw mass ratios from
        #     # The distribution goes as dN/dq = eta_q*q + bb
        #     # Find the bb that normalizes the distribution
        #     bb = (1. - (eta_q/2.)*((qmax**2.) - (qmin**2.)))/(qmax-qmin)

        #     # This is the random numbers between 0 and 1, they will be used for when inverting the CDF to get the q
        #     U = np.random.random(nbr_bin)

        #     # Now solving the pq formel
        #     DD = -(eta_q/2.)*(qmin**2.) - bb*qmin - U
        #     EE = (2./eta_q)*DD
        #     q_minus = -(bb/eta_q) - np.sqrt((bb/eta_q)**2. - EE)
        #     q_plus = -(bb/eta_q) + np.sqrt((bb/eta_q)**2. - EE)
        #     q = np.zeros(nbr_bin)
        #     ind_qminus = (q_minus < qmax)*(q_minus > qmin)*(((q_plus < qmin)+(q_plus > qmax))>0.)
        #     ind_qplus = (q_plus < qmax)*(q_plus > qmin)*(((q_minus < qmin)+(q_minus > qmax))>0.)
        #     q[ind_qminus] = q_minus[ind_qminus]
        #     q[ind_qplus] = q_plus[ind_qplus]

        #     # Tell the log
            
        #     utils_object.write_log('Assigned masses to the secondaries following a 1D polynomial distribution in mass ratio: dN/dq = '+str(eta_q)+'q + bb. \n')
            


        # # Assign masses to the secondaries
        # m2 = q*m[primary]

        # # Now the total mass is too high -> scale down again to the total mass we should have.
        # Mtot1 = np.sum(m)
        # Mtot2 = np.sum(m2)
        # Mtot = Mtot1 + Mtot2
        # n = 0
        # n2 = 0
        # while Mtot > Target_M:
        #     Mtot1 = Mtot1-m[n]
        #     if primary[n]: 
        #         Mtot2 = Mtot2 - m2[n2]
        #         n2 = n2+1
        #     n = n+1
        #     Mtot = Mtot1+Mtot2
        # m = m[n:]
        # primary = primary[n:]
        # single = single[n:]
        # birthday = birthday[n:]
        # # Total number of binary stars
        # nbr_bin = np.sum(primary)
        # # Total number of stars (counting binaries as 1 star)
        # nbr_stars = len(m)

        # # Update initial masses of primaries (m1) and secondaries (m2), initial mass ratio (q)
        # m1 = m[primary]
        # m2 = m2[n2:]
        # q = q[n2:]
        # # Set the birthday of the secondary to the same as the primary
        # birthday_m2 = copy.copy(birthday[primary])

        # # Tell the log
        
        # utils_object.write_log('Removed some random stars so that the total mass is what we want. \n')
        # utils_object.write_log('Total mass: '+str(Mtot)+' MSun, '+str(nbr_bin)+' binaries \n')
        # utils_object.write_log('Set the birthday of the secondaries to the same as the primaries \n')
        


        # if iii == 0:
        #     if save_figs:
        #         # # # # Plot the q # # # # # # # #
        #         fig, ax = plt.subplots(1,1,figsize=(6,4.5))
        #         ax.hist(q,100)
        #         ax.set_xlabel('Mass ratio, $q$')
        #         ax.set_ylabel('Number stars')
        #         ax.set_yticks([])
        #         xtick = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        #         ax.set_xticks(xtick)
        #         ax.set_xticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        #         for i in range(len(xtick)):
        #             ax.get_xaxis().majorTicks[i].set_pad(7)
        #         ax.tick_params('both', length=8, width=1.5, which='major')
        #         ax.tick_params('both', length=4, width=1.0, which='minor')
        #         ax.tick_params(direction="in", which='both')
        #         fig.savefig(plots_dir+'/q.png',format='png',bbox_inches='tight',pad_inches=0.1)
        #         plt.close(fig)
        #         # # # # # # # # # # # # # # # # # # 


        #         # # # Check the IMF again! # # #
        #         fig, ax1 = plt.subplots(1,1,figsize=(8,6))
        #         ax1.hist(np.log10(np.concatenate([m,m2])),100,log=True,label='IMF including M2')
        #         ax1.hist(np.log10(m),100,log=True,color='r',alpha=0.5,label='IMF excluding M2')
        #         ax1.set_xlabel('$\\log_{10} (M/M_{\\odot})$')
        #         ax1.set_ylabel('Number of stars')
        #         ax1.tick_params(direction="in", which='both')
        #         ax1.legend(loc=0,fontsize=0.8*fsize)
        #         fig.savefig(plots_dir+'/IMF_update_after_q.png',format='png',bbox_inches='tight',pad_inches=0.1)
        #         plt.close(fig)
        #         # # # # # # # # # # # # # # # # #




            