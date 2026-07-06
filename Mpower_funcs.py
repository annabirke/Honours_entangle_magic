import numpy as np;
import pandas as pd
import matplotlib.pyplot as plt;
from scipy.linalg import expm
from scipy.signal import find_peaks
import toymodel_funcs


def M2(psiout):
    # for SRE
    traces=[]

    # Pauli spin operators
    I=np.array([[1,0],[0,1]])
    X=np.array([[0,1],[1,0]])
    Y=np.array([[0,-1j],[1j,0]])
    Z=np.array([[1,0],[0,-1]])
    pauli=np.array([I,X,Y,Z])

    # 2 particle density matrix
    rho=np.outer(psiout,psiout.conj())
    
    for i in range(4):
        for j in range(4):
            P=np.kron(pauli[i],pauli[j])
            trace=np.trace(rho@P)**4
            traces.append(trace)
    xi2=1/4*np.sum(traces)
    xi2=complex(xi2.real, 0 if abs(xi2.imag) < 1e-10 else xi2.imag) # removes tiny imaginary components which make the calculations hard
    M2=-np.log(xi2)
    chopped_M2=complex(M2.real, 0 if abs(M2.imag) < 1e-10 else M2.imag) # removes tiny imaginary components which make the calculations hard
    return chopped_M2

def Mpower_inout(a,mu_list,t_list,psi,ylim,savetrue,savename,colourline,mode="in"):
    plt.figure(figsize=(10,4))
    magic_list_mu=[]
    for k, mu in enumerate(mu_list) :
        magic_list = []
        for n, t in enumerate(t_list) :
            if mode == "in":
                psiout = toymodel_funcs.Evolve(a, mu, t) @ psi
                magic = M2(psiout)
                # print(f'imag magic {abs(magic.imag):.3g}')
                magic=complex(magic.real, 0 if abs(magic.imag) < 1e-10 else magic.imag) # removes tiny imaginary components 
            
            elif mode == "out":
                # now check if psiout is a 4dim vector, or already a 4x1000 object for each timestep 
                if psi.ndim == 2:
                    psi_t = psi[:, n] # this makes sure we only take the correct slice of psiout, ie at the correct timestep
                else:
                    psi_t = psi
                
                magic = M2(psi_t) 
                magic=complex(magic.real, 0 if abs(magic.imag) < 1e-10 else magic.imag) # removes tiny imaginary components 
                
            else:
                raise ValueError("mode must be 'in' or 'out'")
            
            magic_list.append(magic)
            
        plt.plot(t_list,magic_list,color=f'{colourline}',label=rf"$\mu=${mu}")
        magic_list_mu.append(magic_list)
    
    plt.axhline(y=np.log(16/7),c='k',ls='--',label='Maximum log(16/7)') # maximal possible magic for two-qubit state, from Liu2026
    plt.xlabel('Time',fontsize=11)
    plt.ylabel(f'Magic power of state',fontsize=11)
    plt.ylim(ylim)
    if mode == "in":
        plt.title(rf'$\psi_{{in}}=$[{psi[0]:.3g}, {psi[1]:.3g}, {psi[2]:.3g}, {psi[3]:.3g}]')
    elif mode == "out":
        plt.title(rf'Input $\psi_{{out}}$')
    if savetrue: 
        plt.savefig(f'C:/Users/annas/Documents/2026/Honours/Entangle_Magic/{savename}',bbox_inches='tight',dpi=300)
    # plt.show()
    return np.array(magic_list_mu)



def ss():
    ss = np.array([
    # 36 product stabilizer states
    # Z ⊗ Z
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1],
    # Z ⊗ X
    [1/np.sqrt(2),  1/np.sqrt(2), 0, 0],
    [1/np.sqrt(2), -1/np.sqrt(2), 0, 0],
    [0,0, 1/np.sqrt(2),  1/np.sqrt(2)],
    [0,0, 1/np.sqrt(2), -1/np.sqrt(2)],
    # Z ⊗ Y
    [1/np.sqrt(2),  1j/np.sqrt(2), 0, 0],
    [1/np.sqrt(2), -1j/np.sqrt(2), 0, 0],
    [0,0, 1/np.sqrt(2),  1j/np.sqrt(2)],
    [0,0, 1/np.sqrt(2), -1j/np.sqrt(2)],
    # X ⊗ Z
    [1/np.sqrt(2),0, 1/np.sqrt(2),0],
    [1/np.sqrt(2),0,-1/np.sqrt(2),0],
    [0,1/np.sqrt(2),0, 1/np.sqrt(2)],
    [0,1/np.sqrt(2),0,-1/np.sqrt(2)],
    # X ⊗ X
    [1/2, 1/2, 1/2, 1/2],
    [1/2,-1/2, 1/2,-1/2],
    [1/2, 1/2,-1/2,-1/2],
    [1/2,-1/2,-1/2, 1/2],
    # X ⊗ Y
    [1/2,  1j/2,  1/2,  1j/2],
    [1/2, -1j/2,  1/2, -1j/2],
    [1/2,  1j/2, -1/2, -1j/2],
    [1/2, -1j/2, -1/2,  1j/2],
    # Y ⊗ Z
    [1/np.sqrt(2),0, 1j/np.sqrt(2),0],
    [1/np.sqrt(2),0,-1j/np.sqrt(2),0],
    [0,1/np.sqrt(2),0, 1j/np.sqrt(2)],
    [0,1/np.sqrt(2),0,-1j/np.sqrt(2)],
    # Y ⊗ X
    [1/2, 1/2,  1j/2,  1j/2],
    [1/2,-1/2,  1j/2, -1j/2],
    [1/2, 1/2, -1j/2, -1j/2],
    [1/2,-1/2, -1j/2,  1j/2],
    # Y ⊗ Y
    [1/2,  1j/2,  1j/2, -1/2],
    [1/2, -1j/2,  1j/2,  1/2],
    [1/2,  1j/2, -1j/2,  1/2],
    [1/2, -1j/2, -1j/2, -1/2],
    
    # 24 entangled stabilizer states
    # Bell basis (real)
    [1/np.sqrt(2),0,0, 1/np.sqrt(2)],
    [1/np.sqrt(2),0,0,-1/np.sqrt(2)],
    [0,1/np.sqrt(2), 1/np.sqrt(2),0],
    [0,1/np.sqrt(2),-1/np.sqrt(2),0],
    # Bell with i phases
    [1/np.sqrt(2),0,0, 1j/np.sqrt(2)],
    [1/np.sqrt(2),0,0,-1j/np.sqrt(2)],
    [0,1/np.sqrt(2), 1j/np.sqrt(2),0],
    [0,1/np.sqrt(2),-1j/np.sqrt(2),0],
    # XX-type entangled
    [1/2, 1/2, 1/2,-1/2],
    [1/2,-1/2, 1/2, 1/2],
    [1/2, 1/2,-1/2, 1/2],
    [1/2,-1/2,-1/2,-1/2],
    # YY-type entangled
    [1/2,  1j/2, -1j/2,  1/2],
    [1/2, -1j/2, -1j/2, -1/2],
    [1/2,  1j/2,  1j/2, -1/2],
    [1/2, -1j/2,  1j/2,  1/2],
    # mixed phase family 1
    [1/2, 1/2,  1j/2, -1j/2],
    [1/2,-1/2,  1j/2,  1j/2],
    [1/2, 1/2, -1j/2,  1j/2],
    [1/2,-1/2, -1j/2, -1j/2],
    # mixed phase family 2
    [1/2,  1j/2,  1/2, -1j/2],
    [1/2, -1j/2,  1/2,  1j/2],
    [1/2,  1j/2, -1/2,  1j/2],
    [1/2, -1j/2, -1/2, -1j/2],
    ], dtype=complex)
    return ss


def Mpower_S(a,mu_list,t_list,ylim,savetrue,savename,colourline):
    # magic power is average magic induced in all stabiliser states
    plt.figure(figsize=(10,4))
    magic_list_mu=[]
    for k, mu in enumerate(mu_list) :
        Mpower_list = []
        for n, t in enumerate(t_list) :
            magic_list=[]
            for i in range(60): # because we want the intially non-magic states, so all the stabiliser states
                psiin = ss()[i,:] # ith stabiliser state 
                psiout = toymodel_funcs.Evolve(a,mu,t)@psiin
                magic_list.append(M2(psiout))
            
            # magic power is average magic induced in all stabiliser states. calculated at a time t
            magicpower = sum(np.array(magic_list))/60 # this is avg magic power for a particular time and particular mu
            Mpower_list.append(magicpower)
            
        plt.plot(t_list,Mpower_list,color=f'{colourline}',label=rf"$\mu=${mu}")
        magic_list_mu.append(Mpower_list)
    
    plt.axhline(y=np.log(16/7),c='k',ls='--',label='Maximum log(16/7)') # maximal possible magic for two-qubit state, from Liu2026
    plt.xlabel('Time',fontsize=11)
    plt.ylabel(rf'Magic power of $\hat{{S}}$',fontsize=11)
    plt.ylim(ylim)
    if savetrue:
        plt.savefig(f'C:/Users/annas/Documents/2026/Honours/Entangle_Magic/{savename}',bbox_inches='tight',dpi=300)
    plt.show()

    return np.array(magic_list_mu)
