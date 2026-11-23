import numpy as np
from itertools import product


# def M2_additive(Psi):
#     # Pauli spin operators
#     I=np.array([[1,0],[0,1]])
#     X=np.array([[0,1],[1,0]])
#     Y=np.array([[0,-1j],[1j,0]])
#     Z=np.array([[1,0],[0,-1]])
#     pauli=np.array([I,X,Y,Z])

#     Psifull = 
#     M2add = np.zeros((1,),dtype=complex)
#     for psi in Psifull:
#         Xi = np.zeros((1,),dtype=complex)
#         for P in pauli:
#             Xi += (psi.conj() @ P @ psi)**4 /4
#         Xi = complex(Xi.real, 0 if abs(Xi.imag) < 1e-10 else Xi.imag)
#         M2 = -np.log(Xi)
#         M2add += complex(M2.real, 0 if abs(M2.imag) < 1e-10 else M2.imag)


def M2_anyP(P,Psi):
    I=np.array([[1,0],[0,1]])
    X=np.array([[0,1],[1,0]])
    Y=np.array([[0,-1j],[1j,0]])
    Z=np.array([[1,0],[0,-1]])
    Pauli=np.array([I,X,Y,Z],dtype=complex)

    Xi = np.zeros((1,),dtype=complex)
    for conf in product(range(4), repeat=P): # gives all the combinations of indices 0 to 3 for P particles
        Pauli_list = np.array([Pauli[conf[i]] for i in range(P)]) # array of the pauli matrices for each particle based on the configuration
        # now we kronecker product them all together
        Pstring = Pauli_list[0]
        for Pi in Pauli_list[1:]:
            Pstring = np.kron(Pstring,Pi)
        # print('Pstring shape:',Pstring.shape)
        
        # we now have a Pauli string. Next need to find Xi for this string and add it up
        Xi += 2**(-P)*(Psi.conj() @ Pstring @ Psi)**4 
    
    Xi = complex(Xi.real, 0 if abs(Xi.imag) < 1e-10 else Xi.imag)
    M2 = -np.log(Xi)
    M2 = complex(M2.real, 0 if abs(M2.imag) < 1e-10 else M2.imag)
    return M2

