import numpy as np;
import pandas as pd
import matplotlib.pyplot as plt;
from scipy.linalg import expm
from scipy.signal import find_peaks

def Evolve(a,mu,t):
    H = np.array([[a+mu,-a/2,-a/2,0],
                  [-a/2,a,0,-a/2],
                  [-a/2,0,a,-a/2],
                  [0,-a/2,-a/2,a+mu]])
    U = expm(-1j*H*t)
    return U



def N_L(a,mu_list,t_list,psi,savetrue,savename,mode="in"):
    NL = np.array([[2,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,0]])
    plt.figure(figsize=(10,4))
    exNL_mu=[]
    for k, mu in enumerate(mu_list):
        exNL=[]
        for n, t in enumerate(t_list):
            if mode == "in":
                psiout = Evolve(a,mu,t) @ psi
                exNL.append(psiout.conj().T @ NL @ psiout)
                
            elif mode == "out":
                if psi.ndim == 2:
                    psiout = psi[:, n] # this makes sure we only take the correct slice of psiout, ie at the correct timestep
                else:
                    raise ValueError("if mode 'out' need psi already as 4x1000 array for each time in tlist")
                
                exNL.append(psiout.conj().T @ NL @ psiout)
        
        exNL=np.array(exNL)
        plt.plot(t_list,exNL,color='mediumvioletred',label=rf'$\mu=${mu}')
        
        exNL_mu.append(exNL)

    plt.ylabel(fr'$\langle N_L \rangle$',fontsize=15)
    plt.xlabel('Time',fontsize=15)
    if savetrue:
        plt.savefig(f'C:/Users/annas/Documents/2026/Honours/Entangle_Magic/{savename}',bbox_inches='tight',dpi=300)
    # plt.show()
    
    return np.array(exNL_mu)
