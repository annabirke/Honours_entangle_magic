# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

This is a single-notebook research codebase (Honours physics project), not a
package. All code lives in [2_slater_23August.ipynb](2_slater_23August.ipynb).
There is no build system, test suite, linter, or `requirements.txt` — the
notebook is run interactively in Jupyter/VSCode against an Anaconda Python
environment (numpy, scipy, matplotlib, pandas).

To run it: open the notebook in Jupyter or VSCode and execute cells top to
bottom. Cells have hidden state dependencies (later cells reference `psi0`,
`y0`, `solution`, `f_soln`, `psi_soln` defined earlier) so partial/out-of-order
execution will raise `NameError`.

## important note
Do not remove or change any comments I have added to the code, unless that section of code gets completely changed, in which case ask me first and explicitly say you will change my comment. 

## Physics/architecture

The notebook implements a **multi-configuration time-dependent variational
method**: the many-body wavefunction is a superposition of `N` Slater
determinants, each built from its own set of time-dependent single-particle
orbitals (the orbitals are *not* orthogonal across determinants, so the
Slater determinants themselves are non-orthogonal — most of the code
complexity comes from tracking these overlaps).

**Core objects**

- `P` — number of particles; `N` — number of Slater determinants in the
  superposition.
- `psi` — complex array, shape `(N, P, 2)`. `psi[gamma, k, :]` is particle
  `k`'s single-particle orbital within determinant `gamma`, expressed in a
  2-dimensional single-particle basis (a two-site L/R model).
- `f` — complex array, shape `(N,)`. Weight of determinant `gamma` in the
  full superposition (does *not* need `|f|=1`; only the full state needs to
  be normalized once cross-terms from non-orthogonal determinants are
  included).
- `y` — the flattened state vector for ODE integration: `y = [f, psi.ravel()]`,
  shape `(N + N*P*2,)`.
- `Ham` — the full many-body Hamiltonian (`2^P x 2^P`), built from a
  single-particle hopping term `h0` (amplitude `a`) and an on-site
  interaction `v` (strength `mu`) via Kronecker products.

**Overlap machinery** (`overlap`, `N_`, `N_p`, `N_kp`) — because
determinants are non-orthogonal, matrix elements between determinants
factorize into products of single-particle overlaps. `N_p` / `N_kp` compute
that product with one/two particle slots excluded — needed when
differentiating a determinant overlap with respect to one orbital.

**Hamiltonian matrix elements** (`H`, `H_p_xp_yp`) — `H(psi)` builds the
`N x N` matrix `<Phi_gamma|Ham|Phi_beta>` by taking the tensor product of
single-particle orbitals to form full many-body kets/bras. `H_p_xp_yp`
builds the analogous matrix element with particle `p`'s orbital replaced by
basis states `xp`/`yp` — used to construct the single-particle equations of
motion.

**Equations of motion** — from the time-dependent variational principle, the
EOM for `f` and `psi` are coupled and implicit: schematically
`I_all @ d_y = M_all @ d_y + base_all`, solved each RHS evaluation as
`d_y = solve(I_all - M_all, base_all)` inside `coupled(t, y)` (the function
passed to `solve_ivp`). The block structure is:
- `I_f`/`I_psi` — overlap-derived matrices on the LHS.
- `M_fpsi`/`M_psif`/`M_psipsi` — cross-coupling terms between `d_f` and
  `d_psi`.
- `df_base`/`dpsi_base` — the inhomogeneous (Hamiltonian-driven) part.

`conditionno(t, y)` mirrors `coupled` but returns `cond(I_all - M_all)`
instead of solving.

### Known issues in the notebook (diagnosed, fixed in `tdvp_multislater.py`)

The notebook's `solve_ivp` call stalls. Two independent causes, both
confirmed numerically:

1. **`H_p_xp_yp` never replaces the `p == 0` slot.** The tensor-product
   loop starts at `k = 1` with slot 0 pre-loaded from `psi`, so the
   `elif k == exclude_p` branch cannot fire for `exclude_p == 0`. This
   makes the linear system *inconsistent* — at the notebook's initial
   condition the least-squares residual is 2.25 against `||b|| = 6.59`,
   i.e. no solution exists. `np.linalg.solve` does not report this.

2. **`I_all - M_all` is exactly singular, not near-singular.** Measured
   rank is 4 of 10 (nullity 6, cond ~3e17), *independent of initial
   condition* — so the `conditionno` scan over `th`/`f0` cannot help. The
   null space is pure gauge: rescaling `phi[gamma,k] -> (1+c)phi[gamma,k]`
   with `f[gamma] -> (1-sum_k c)f[gamma]` leaves `|Psi>` invariant, giving
   `N*P` exact null directions; for `N=P=2, d=2` there are 2 more because
   the ansatz saturates the Hilbert space (the CP/PARAFAC form is
   non-unique). The system is consistent but underdetermined — every
   solution gives the same physical trajectory — so the fix is a
   rank-truncated pseudo-inverse (minimum-norm solution) rather than
   `np.linalg.solve`, optionally plus the MCTDH gauge
   `<phi|d_t phi> = 0` to pin orbital norms.

Also note `assert Ham.shape == (2**N, 2**N)` should be `2**P` (passes only
because `N == P == 2` here), and the energy-conservation cell einsums
against `H` (the *function*) rather than `Ham`.

### `tdvp_multislater.py`

Standalone module with both fixes. `build_linear_system` assembles the same
two boxed equations (vectorised via einsum), `minnorm_solve` does the
SVD-truncated solve, `integrate` wraps `solve_ivp`, and `exact_state` gives
the `expm(-i H t)` benchmark. For `N = P = d = 2` the ansatz spans the whole
Hilbert space, so TDVP must reproduce the exact dynamics — `main()` checks
this and currently agrees to 3e-10 with norm/energy conserved at the
integrator tolerance floor.

**Integration & diagnostics** — `solve_ivp` (DOP853, `rtol=1e-9`,
`atol=1e-11`) integrates `coupled` over `y0`. Downstream cells validate the
result:
- Slater-determinant overlap `<Phi^0|Phi^1>(t)` over time.
- Full-state normalization `<Psi|Psi>(t)` (should be conserved but isn't
  exactly, per the code's own comment on the energy/norm cell).
- Energy conservation `<Psi|H|Psi>(t)`, and that quantity divided by
  `<Psi|Psi>(t)` (compared against the TDGCM-paper convention).
- Single-particle orbital normalization per determinant/particle.
- An alternate re-derivation of the determinant-overlap ODE via numerical
  `d(log f0)/dt` interpolation, integrated independently as a cross-check.

When editing the EOM derivation, keep the index convention consistent
throughout: `gamma`/`beta` index Slater determinants, `k`/`p` index
particles, `xk`/`xp`/`yp` index the 2-dim single-particle basis — the
reshape/flatten order (`(N, P, 2)` → `N*P*2`) is relied on implicitly when
building `M_*` and `I_*` matrices via `.reshape(...)`.
