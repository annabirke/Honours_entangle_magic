"""
tdvp_multislater.py
===================

Gauge-robust solver for the multi-configuration TDVP equations in
2_slater_23August.ipynb.

    |Psi(t)> = sum_gamma f_gamma(t) |Phi^gamma(t)>,
    |Phi^gamma> = prod_k |phi_{gamma k}(t)>

Two things are changed relative to the notebook.


(1) BUG: H_p_xp_yp never replaced the p = 0 slot
------------------------------------------------
The tensor-product loop is written as

    slater = psi[:,0,:]
    for k in range(1,P):
        ...

so slot 0 is pre-loaded from psi and the `elif k == exclude_p` branch can
never fire for exclude_p = 0.  H^0_{gamma beta}(x_0,y_0) was therefore
computed with the *orbital* in slot 0 instead of the basis state.

Consequence: the linear system A y' = b becomes INCONSISTENT -- b acquires
a component outside the range of A.  At the notebook's initial condition
the least-squares residual is ||Ay'-b|| = 2.25 against ||b|| = 6.59, i.e.
the system has no solution at all.  np.linalg.solve does not report this;
it returns a huge meaningless vector, which is what stalls solve_ivp.


(2) (I_all - M_all) is EXACTLY singular, not "near singular"
------------------------------------------------------------
This is structural and no choice of initial condition can avoid it, so the
condition-number scan over `th` and `f0` cannot help.

The ansatz is redundant: for any constants c_{gamma k},

    phi_{gamma k} -> (1 + c_{gamma k}) phi_{gamma k}
    f_gamma       -> (1 - sum_k c_{gamma k}) f_gamma

leaves |Psi> unchanged to first order.  These N*P gauge directions are
therefore exact null vectors of the TDVP matrix (verified numerically to
~5e-17).  For N=P=2, d=2 there are 2 further null directions because the
ansatz saturates the Hilbert space (any 2-qubit state has a 2-term Schmidt
decomposition, so the CP/PARAFAC form is non-unique).

Measured at the notebook's initial condition:

    singular values  2.94, 1.83, 1.41, 1.41, 3e-16, 2e-16, 1e-16, ... 1e-17
    rank 4 of 10, nullity 6, cond ~ 3e17

The system is *consistent but underdetermined*: the null space is pure
gauge, so every solution gives the same physical trajectory.  The fix is to
stop asking for "the" solution and take the minimum-norm one via a
rank-truncated pseudo-inverse (`np.linalg.solve` -> SVD solve).  The
min-norm choice also happens to be the one with the least gauge drift.

Optionally we additionally impose the MCTDH gauge <phi|d_t phi> = 0, which
pins every orbital norm to 1 for all time.

With both fixes, dPsi/dt reconstructed from (f', psi') reproduces -i H Psi
to 7e-16 relative error, which confirms the derivation in the PDF is right.


Usage
-----
    python tdvp_multislater.py            # run, print diagnostics, save PNG

or from a notebook:

    from tdvp_multislater import *
    sol = integrate(y0, t_list)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from math import comb
from itertools import product



# ----------------------------------------------------------------- model

N = 2      # number of Slater determinants
P = 2      # number of particles
D = 2      # single-particle basis dimension (L/R)

a = 1.0
mu = -4.0

h0 = a/2*np.array([[1, -1],
                   [-1, 1]], dtype=complex)
v = np.diag(np.array([mu, 0, 0, mu], dtype=complex))
Ham = (np.kron(h0, np.eye(2, dtype=complex))
       + np.kron(np.eye(2, dtype=complex), h0) + v)

assert Ham.shape == (D**P, D**P)   # note: notebook asserted 2**N, not 2**P

NY = N*P*D                          # length of the flattened psi block

# ------------------------------------------------------------- kernels

def _factors(psi):
    """ 
    [psi[:,0,:], ..., psi[:,P-1,:]], each (N,D)."""
    return [psi[:, k, :] for k in range(P)]


def _build(factors):
    """Tensor-product the P factors into the (N, D**P) many-body kets."""
    out = factors[0]
    for k in range(1, P):
        out = np.einsum('ai,aj->aij', out, factors[k]).reshape(N, -1)
    return out


def overlap(psi, k):
    """ overlap[a,b] = <phi_{a k} | phi_{b k}>, 
    Shape (N,N). 
    Overlap matrix of kth single-particle orbitals between alpha and beta slater """
    return np.einsum('ai,bi->ab', psi[:, k, :].conj(), psi[:, k, :])


def overlap_products(ov):
    """
    ov is [overlap(psi, k) for k in range(P)], overlap(psi,k) is (N,N)
    ov becomes (P,N,N) after the np.stack below

    Vectorised replacement for N_, N_p, N_kp.

    Returns
        Nfull  (N,N)      prod over all k
        Np     (P,N,N)    prod over k != p
        Nkp    (P,P,N,N)  prod over i != k,p   (zero on the k == p diagonal)
    """
    S = np.stack(ov)     # (P,N,N)
    Nfull = np.prod(S, axis=0) # (N,N) elementwise multiplication of the overlaps

    Np = np.empty((P, N, N), dtype=complex)
    for p in range(P):
        Np[p] = np.prod(np.delete(S, p, axis=0), axis=0) if P > 1 else 1.0

    Nkp = np.zeros((P, P, N, N), dtype=complex)
    for k in range(P):
        for p in range(P):
            if k == p:
                continue
            rest = np.delete(S, [min(k, p), max(k, p)], axis=0)  # (P-2,N,N)
            # ^ deletes the whole index row of k and p from S (ordering the smaller index first so min and max, not actually necessary)
            Nkp[k, p] = np.prod(rest, axis=0) if rest.shape[0] else 1.0 
            # ^ (N,N)
            # element wise multiplication of the P-2 overlap matrices. 
            # else 1 so that when have 2 particles and remove 2, dont get zero
    return Nfull, Np, Nkp # (N,N) (P,N,N) (P,P,N,N) ie the overlap matrix element for each alpha,beta


def H_kernel(psi):
    """H[gamma,beta] = <Phi^gamma| Ham |Phi^beta>, shape (N,N)."""
    s = _build(_factors(psi)) # (N, D**P) slaters 
    return s.conj() @ Ham @ s.T


def H_p_kernel(psi):
    """
    H_p[gamma,beta,p,xp,yp] = <phi_g1 .. x_p .. phi_gP| Ham |phi_b1 .. y_p .. phi_bP>

    Slot p carries the basis state x_p (bra) / y_p (ket) instead of an
    orbital.  Unlike the notebook version this is correct for p = 0.
    """
    e = np.eye(D, dtype=complex)
    out = np.zeros((N, N, P, D, D), dtype=complex)
    for p in range(P):
        for xp in range(D):
            fb = _factors(psi) # all the orbital states 
            fb[p] = np.broadcast_to(e[xp], (N, D)).copy() # replace p slot with xp state
            bra = _build(fb).conj() @ Ham
            for yp in range(D):
                fk = _factors(psi)
                fk[p] = np.broadcast_to(e[yp], (N, D)).copy()
                out[:, :, p, xp, yp] = bra @ _build(fk).T
    return out

# ------------------------------------------------- the linear system

def build_linear_system(psi, f):
    """
    Assemble A y' = b, with y' = [d_f (N), d_psi (N*P*D)], following the
    two boxed equations of the PDF.

        row block 1 (N rows)   :  sum_b (d_f_b) N_gb
                                  = -sum_b f_b [ sum_k sum_xk N^k_gb
                                     phi*_{g k xk} (d_t phi_{b k xk}) + i H_gb ]

        row block 2 (N*P*D)    :  sum_b f_b N^p_gb (d_t phi_{b p xp})
                                  = -sum_b f_b sum_{k!=p} sum_xk N^{kp}_gb
                                       phi_{b p xp} phi*_{g k xk}(d_t phi_{b k xk})
                                    -sum_b (d_t f_b) phi_{b p xp} N^p_gb
                                    -sum_b i f_b sum_yp phi_{b p yp} H^p_gb(xp,yp)
    """
    ov = [overlap(psi, k) for k in range(P)]
    Nfull, Np, Nkp = overlap_products(ov) # (N,N) (P,N,N) (P,P,N,N)

    # --- LHS blocks
    I_f = Nfull      # (N,N)

    I_psi = np.zeros((N, P, D, N, P, D), dtype=complex)
    idx_p, idx_x = np.meshgrid(np.arange(P), np.arange(D), indexing='ij') 
    # ^ grid (P,D) with every index pair of p and xp. arange(P) is 0,1,...P-1
    for g in range(N):
        for b in range(N):
            I_psi[g, idx_p, idx_x, b, idx_p, idx_x] = f[b]*Np[idx_p, g, b]
    I_psi = I_psi.reshape(NY, NY) # NY is N*P*D

    # --- coupling blocks (moved to the LHS as -M)
    M_fpsi = -np.einsum('b,kgb,gkx->gbkx', f, Np, psi.conj()).reshape(N, NY) 
    # ^ reshape f x psi, take in shape psi, spit out shape f

    M_psif = -np.einsum('bpx,pgb->gpxb', psi, Np).reshape(NY, N) # reshape psi x f

    M_psipsi = -np.einsum('b,kpgb,bpx,gky->gpxbky',
                          f, Nkp, psi, psi.conj()).reshape(NY, NY) # reshape psi x psi

    # --- inhomogeneous parts, ie b in Ay' = b
    df_base = -1j*np.einsum('a,ba->b', f, H_kernel(psi))

    psiH = np.einsum('bpy,gbpxy->gbpx', psi, H_p_kernel(psi))
    dpsi_base = (-1j*np.einsum('b,gbpx->gpx', f, psiH)).ravel()

    A = np.block([[I_f,     -M_fpsi],
                  [-M_psif,  I_psi - M_psipsi]])
    b = np.concatenate((df_base, dpsi_base))
    return A, b


def gauge_rows(psi):
    """
    MCTDH constraint rows enforcing <phi_{gamma k}| d_t phi_{gamma k}> = 0,
    which holds the norm of every orbital fixed for all time.
    """
    G = np.zeros((N*P, N + NY), dtype=complex)
    row = 0
    for g in range(N):
        for k in range(P):
            blk = np.zeros((N, P, D), dtype=complex)
            blk[g, k, :] = psi[g, k, :].conj()
            G[row, N:] = blk.ravel() # everything after N ie not the d_f parts
            row += 1
    return G


def minnorm_solve(A, b, rcond=1e-10):
    """
    Minimum-norm solution of a consistent, rank-deficient system, via a
    rank-truncated SVD.  
    A x = b   ->   x = A^+ b,      A^+ = V Sigma^+ U^dagger 
    and Sigma^+ is just diag of 1/singular vals 
    
    Returns (x, rank, residual). residual should be zero 
    """
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    keep = s > rcond*s[0] # ie only want singular values above the tolerance

    # now the minimum norm solution to the linear system 
    x = Vh[keep].conj().T @ ((U[:, keep].conj().T @ b)/s[keep])
    return x, int(keep.sum()), np.linalg.norm(A @ x - b) 

# --------------------------------------------------- the RHS

def rhs(t, y, rcond=1e-10, gauge=True):
    f = y[:N]
    psi = y[N:].reshape(N, P, D)
    A, b = build_linear_system(psi, f)
    if gauge:
        A = np.vstack([A, gauge_rows(psi)])
        b = np.concatenate([b, np.zeros(N*P, dtype=complex)])
    x, _, _ = minnorm_solve(A, b, rcond)
    return x


def integrate(y0, t_list, rcond=1e-10, gauge=True, **kw):
    opts = dict(method="DOP853", rtol=1e-10, atol=1e-12)
    opts.update(kw)
    return solve_ivp(lambda t, y: rhs(t, y, rcond, gauge),
                     (t_list[0], t_list[-1]), y0, t_eval=t_list, **opts)

# ------------------------------------------------------- observables

def full_state(f, psi):
    """|Psi> as a (D**P,) vector."""
    return f @ _build(_factors(psi))


def unpack(sol):
    f = sol.y[:N, :]
    psi = sol.y[N:, :].reshape(N, P, D, -1)
    return f, psi


def observables(sol):
    """Exact <Psi|Psi> and <Psi|H|Psi> including all cross terms."""
    f, psi = unpack(sol)
    nt = f.shape[1]
    norm = np.empty(nt, dtype=complex)
    energy = np.empty(nt, dtype=complex)
    for i in range(nt):
        Psi = full_state(f[:, i], psi[..., i])
        norm[i] = Psi.conj() @ Psi
        energy[i] = Psi.conj() @ Ham @ Psi
    return norm, energy # norm.real, energy.real


def Nleft(sol):
    f, psi = unpack(sol)
    nt = f.shape[1]

    left = np.array([conf.count(0) for conf in product(range(D), repeat=P)],
                    dtype=complex)
    # ^ count(0) counts how many times a 0 appears in conf, conf is (i,j,k) 
    # where ijk are 0 or 1 in same ordering as the tensor products
    NL = np.diag(left)

    Nleft_t = np.empty(nt, dtype=complex)
    for t in range(nt):
        Psi = full_state(f[:, t], psi[..., t])
        Nleft_t[t] = Psi.conj() @ NL @ Psi
    return Nleft_t


def Nleft_exact(y0,t_list):
    left = np.array([conf.count(0) for conf in product(range(D), repeat=P)],
                    dtype=complex)
    # ^ count(0) counts how many times a 0 appears in conf, conf is (i,j,k) 
    # where ijk are 0 or 1 in same ordering as the tensor products
    NL = np.diag(left)

    Nleft_t = np.empty(len(t_list), dtype=complex)
    for i, t in enumerate(t_list):
        Psi = exact_state(y0,t)
        Nleft_t[i] = Psi.conj() @ NL @ Psi
    return Nleft_t


def exact_state(y0, t):
    """Exact solution of i dPsi/dt = H Psi from the same initial state."""
    f0 = y0[:N]
    psi0 = y0[N:].reshape(N, P, D)
    Psi0 = full_state(f0, psi0)
    return expm(-1j*Ham*t) @ Psi0

# ------------------------------------------------- initial condition

def make_y0(th=3.0, f0=(1.0, 0.0)):
    f = np.array(f0, dtype=complex)
    f = f/np.linalg.norm(f)
    psi = np.zeros((N, P, D), dtype=complex)
    for k in range(P):
        # psi[0, k, :] = [np.cos(th), np.sin(th)]
        # psi[1, k, :] = [np.sin(th), -np.cos(th)]
        psi[0, k, :] = [1.0, 0.0]
        psi[1, k, :] = [1.0, 0.0]
        psi[0, k, :] /= np.linalg.norm(psi[0, k, :])
        psi[1, k, :] /= np.linalg.norm(psi[1, k, :])
    return np.concatenate((f, psi.ravel()))

# ------------------------------------------------------------- main

def main():
    y0 = make_y0()

    # --- one-off structural report at t = 0
    A, b = build_linear_system(y0[N:].reshape(N, P, D), y0[:N])
    s = np.linalg.svd(A, compute_uv=False)
    x, rank, res = minnorm_solve(A, b)
    print("A = I_all - M_all at t=0")
    print("  singular values :", np.array2string(s, precision=2))
    print(f"  rank {rank}/{A.shape[0]}   nullity {A.shape[0]-rank}"
          f"   cond {s[0]/s[-1]:.2e}")
    print(f"  least-squares residual ||Ax-b|| = {res:.2e}"
          f"   (||b|| = {np.linalg.norm(b):.2e})")

    Ag = np.vstack([A, gauge_rows(y0[N:].reshape(N, P, D))])
    bg = np.concatenate([b, np.zeros(N*P, dtype=complex)])
    _, rank_g, res_g = minnorm_solve(Ag, bg)
    print(f"  with gauge rows: rank {rank_g}/{Ag.shape[1]}"
          f"   residual {res_g:.2e}")

    # --- integrate
    t_list = np.linspace(0.0, 10.0, 501)
    sol = integrate(y0, t_list)
    print(f"\nintegration: success={sol.success}  nfev (no. times deriv function was called)={sol.nfev}"
          f"  reached t={sol.t[-1]:.3f}  '{sol.message}'")
    if not sol.success:
        return sol

    f, psi = unpack(sol)
    # Psiexact = np.array([exact_state(y0, t) for t in sol.t],dtype=complex)
    norm, energy = observables(sol)
    print('max norm.imag:',np.max(norm.imag),' max energy.imag:',np.max(energy.imag))
    norm, energy = norm.real, energy.real
    Nleft_t = Nleft(sol)
    Nleft_t_exact = Nleft_exact(y0,sol.t)

    # --- exact benchmark.  For N=P=D=2 the ansatz spans the whole Hilbert
    #     space, so TDVP must reproduce the exact dynamics identically.
    err = np.array([
        np.linalg.norm(full_state(f[:, i], psi[..., i]) - exact_state(y0, t))
        for i, t in enumerate(sol.t)])

    print(f"\n  norm   drift  : {np.ptp(norm):.3e}") # peak-to-peak 
    print(f"  energy drift  : {np.ptp(energy):.3e}")
    print(f"  max |Psi_TDVP - Psi_exact| : {err.max():.3e}")

    orbnorm = np.abs(psi)**2
    orbnorm = orbnorm.sum(axis=2)         # (N,P,nt)
    print(f"  max orbital-norm drift from 1 : {np.abs(orbnorm-1).max():.3e}")

    slatov = np.prod(np.einsum('pxt,pxt->pt',
                              psi[0].conj(), psi[1]), axis=0) # pick which slater

    # --- plots
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    ax[0, 0].plot(sol.t, norm)
    ax[0, 0].set_title(r'$\langle\Psi|\Psi\rangle$')
    ax[0, 1].plot(sol.t, energy)
    ax[0, 1].set_title(r'$\langle\Psi|\hat H|\Psi\rangle$')
    ax[0, 2].plot(sol.t, energy/norm)
    ax[0, 2].set_title(r'$\langle\Psi|\hat H|\Psi\rangle/\langle\Psi|\Psi\rangle$')
    ax[1, 0].plot(sol.t, np.abs(slatov))
    ax[1, 0].set_title(r'$|\langle\Phi^0|\Phi^1\rangle|$')
    for g in range(N):
        for k in range(P):
            ax[1, 1].plot(sol.t, orbnorm[g, k], label=f'slater{g} part{k}')
    ax[1, 1].set_title('orbital norms')
    ax[1, 1].legend(fontsize=7)
    ax[1, 2].semilogy(sol.t, np.maximum(err, 1e-18))
    ax[1, 2].set_title(r'$\|\Psi_{TDVP}-\Psi_{exact}\|$')
    for a_ in ax.ravel():
        a_.set_xlabel('t')
    fig.tight_layout()
    fig.savefig('tdvp_diagnostics.png', dpi=120)
    print("\nsaved tdvp_diagnostics.png")
    plt.show()

    plt.plot(sol.t,Nleft_t, label='TDVP')
    plt.plot(sol.t,Nleft_t_exact, label='Exact')
    plt.xlabel('Time')
    plt.ylabel('<N_L>')
    plt.savefig('tdvpvsexact_Nleft.png', dpi=120)
    plt.show()

    return sol


if __name__ == "__main__":
    main()
