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




''' NONLOCAL MAGIC FOR 2 QUBITS '''
from functools import reduce

# --- Pauli strings -----------------------------------------------------------

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULIS_1Q = [_I, _X, _Y, _Z]


def pauli_strings(n_qubits):
    """All 4^n Pauli strings on n qubits, as a (4^n, 2^n, 2^n) array."""
    return np.array([reduce(np.kron, ops)
                     for ops in product(_PAULIS_1Q, repeat=n_qubits)])


_P2 = pauli_strings(2)  # 16 two-qubit Pauli strings


# --- Helpers -----------------------------------------------------------------

def _as_series(psi):
    """
    Return (psi as shape (4, T), normalised; was_single_state).
    Time axis last.
    """
    psi = np.asarray(psi, dtype=complex)
    single = psi.ndim == 1
    if single:
        psi = psi[:, None]                      # (4,) -> (4, 1)
    if psi.shape[0] != 4:
        raise ValueError(f"expected shape (4,) or (4, T), got {psi.shape}")
    psi = psi / np.linalg.norm(psi, axis=0, keepdims=True)
    return psi, single


def _out(x, single):
    """Drop the trailing length-1 time axis for single-state input."""
    return x[..., 0] if single else x


# --- Stabiliser Renyi entropy M2 ---------------------------------------------

def stabilizer_renyi_2(psi, base=np.e):
    """
    M2(|psi>) = -log( sum_P <psi|P|psi>^4 / d ),   d = 4.
    (Equivalent to the paper's Eq. (D1).)
    psi: shape (4,) or (4, T). Returns scalar or (T,).
    """
    psi, single = _as_series(psi)
    d = psi.shape[0]
    # <psi|P|psi> for every Pauli string and time step: shape (16, T)
    expvals = np.einsum('it,pij,jt->pt', psi.conj(), _P2, psi).real
    m2 = -np.log(np.sum(expvals**4, axis=0) / d) / np.log(base)
    m2 = np.maximum(m2, 0.0)  # remove -1e-16 rounding noise (M2 >= 0)
    return _out(m2, single)


# --- Schmidt values ----------------------------------------------------------

def schmidt_values(psi):
    """
    Schmidt values lambda_i (eigenvalues of rho_A), sorted decreasing.
    psi: shape (4,) or (4, T). Returns shape (2,) or (2, T).
    """
    psi, single = _as_series(psi)
    # M[a, b, t] = <ab|psi(t)>; move time first only for the batched SVD
    M = psi.reshape(2, 2, -1).transpose(2, 0, 1)        # (T, 2, 2)
    sv = np.linalg.svd(M, compute_uv=False)              # (T, 2), decreasing
    lam = (sv**2).T                                      # (2, T)
    return _out(lam, single)


# --- Nonlocal magic: the paper's method --------------------------------------

def nonlocal_M2(psi, base=np.e):
    """
    Paper's nonlocal-magic estimate M2({lambda_i}) for a two-qubit state.
    Builds |psi'> = sqrt(l0)|00> + sqrt(l1)|11> and returns M2(|psi'>).
    psi: shape (4,) or (4, T). Returns scalar or (T,).
    """
    psi, single = _as_series(psi)
    lam = schmidt_values(psi)                            # (2, T)
    psi_prime = np.zeros_like(psi)                       # (4, T)
    # construct the state with the equivalent schmidt spectrum ie diagonal matrix
    psi_prime[0] = np.sqrt(lam[0])                       # |00>
    psi_prime[3] = np.sqrt(lam[1])                       # |11>
    return _out(stabilizer_renyi_2(psi_prime, base=base), single)


def nonlocal_M2_closed_form(psi, base=np.e):
    """Same quantity via the two-qubit closed form (cross-check)."""
    psi, single = _as_series(psi)
    lam = schmidt_values(psi)                            # (2, T)
    c = lam[0] - lam[1]
    s = 2 * np.sqrt(np.clip(lam[0] * lam[1], 0, None))
    m = -np.log((1 + c**4 + s**4) / 2) / np.log(base)
    return _out(np.maximum(m, 0.0), single)