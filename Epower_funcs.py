import numpy as np;
import pandas as pd
import matplotlib.pyplot as plt;
from scipy.linalg import expm
from scipy.signal import find_peaks
import toymodel_funcs
import Mpower_funcs

def Epower_inout(a,mu_list,t_list,psi,ylim,savetrue,savename,colourline,mode="in"):
    plt.figure(figsize=(10,4))
    entanglement_list_mu=[]
    for k, mu in enumerate(mu_list) :
        entanglement_list = []
        for n, t in enumerate(t_list) :
            """ 
            According to Beane2019
            get initial state, evolve it, make density matrix for both particles, 
            trace over particle b, square reduced density matrix and trace again to get purity 
            """
            
            if mode == "in":
                psiout = toymodel_funcs.Evolve(a, mu, t) @ psi
            
            elif mode == "out":
                # now check if psiout is a 4dim vector, or already a 4x1000 object for each timestep 
                if psi.ndim == 2:
                    psiout = psi[:, n]
                else:
                    psioutt = psi
                # this makes sure we only take the correct slice of psiout, ie at the correct timestep                
            else:
                raise ValueError("mode must be 'in' or 'out'")
                            
            rhoab = np.outer(psiout, psiout.conj())
            rhoab = rhoab.reshape(2,2,2,2)
            rhoa = np.trace(rhoab,axis1=1,axis2=3) # reduced density matrix for a, traces over second (index=1) and 4th index, which is particle b
            
            purity = np.trace(rhoa @ rhoa)
            chopped_purity = complex(purity.real, 0 if abs(purity.imag) < 1e-10 else purity.imag) # removes tiny imaginary components which make the calculations hard
            if purity.real<0:
                print('neg purity for state i=',i)
            entanglement = 1-chopped_purity 
            entanglement_list.append(entanglement)
            
        plt.plot(t_list,entanglement_list,color=f'{colourline}',label=rf"$\mu=${mu}")
        entanglement_list_mu.append(entanglement_list)
    plt.legend()
    plt.xlabel('Time',fontsize=11)
    plt.ylabel(f'Entanglement power of state',fontsize=11)
    plt.ylim(ylim)
    if mode == "in":
        plt.title(rf'$\psi_{{in}}=$[{psi[0]:.3g}, {psi[1]:.3g}, {psi[2]:.3g}, {psi[3]:.3g}]')
    elif mode == "out":
        plt.title(rf'Input $\psi_{{out}}$')
    if savetrue: 
        plt.savefig(f'C:/Users/annas/Documents/2026/Honours/Entangle_Magic/{savename}',bbox_inches='tight',dpi=300)
    # plt.show()
    return np.array(entanglement_list_mu)



def Epower_S(a,mu_list,t_list,ylim,savetrue,savename,colourline):
    plt.figure(figsize=(10,4))
    entanglement_list_mu=[]
    for k, mu in enumerate(mu_list) :
        entanglement_list = []
        for n, t in enumerate(t_list) :
            """ 
            According to Beane2019
            get initial state, evolve it, make density matrix for both particles, 
            trace over particle b, square reduced density matrix and trace again to get purity 
            """
            purity_list=[]
            
            for i in range(36): # because we want the intially nonentangled states
                psiin = Mpower_funcs.ss()[i,:] # ith state
                psiout = toymodel_funcs.Evolve(a,mu,t) @ psiin
                
                rhoab = np.outer(psiout, psiout.conj())
                # print('rhoab: ',rhoab)
                rhoab = rhoab.reshape(2,2,2,2)
                rhoa = np.trace(rhoab,axis1=1,axis2=3) # reduced density matrix for a, traces over second (index=1) and 4th index which is b
                
                purity = np.trace(rhoa @ rhoa)
                chopped_purity = complex(purity.real, 0 if abs(purity.imag) < 1e-10 else purity.imag) # removes tiny imaginary components which make the calculations hard
                if purity.real<0:
                    print('neg purity for state i=',i)

                purity_list.append(chopped_purity)

            # entanglement power of S is average entanglement of all non-entangled stabiliser states
            entanglement = 1-abs(sum(purity_list))/36 
            entanglement_list.append(entanglement)
            
        plt.plot(t_list,entanglement_list,color=f'{colourline}',label=rf"$\mu=${mu}")
        entanglement_list_mu.append(entanglement_list)
    plt.legend()
    plt.xlabel('Time',fontsize=11)
    plt.ylabel(rf'Entanglement power of $\hat{{S}}$',fontsize=11)
    plt.ylim(ylim)
    if savetrue: 
        plt.savefig(f'C:/Users/annas/Documents/2026/Honours/Entangle_Magic/{savename}',bbox_inches='tight',dpi=300)
    plt.show()
    return np.array(entanglement_list_mu)

