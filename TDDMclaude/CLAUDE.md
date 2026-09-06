# TDDM in the Lipkin model

Honours project: solving the **Time-Dependent Density Matrix (TDDM)** equations for the
Lipkin–Meshkov–Glick model, and comparing against exact diagonalisation and mean-field (TDHF).

Primary references:
- **Tohyama 2020** — TDDM formulation in a time-independent s.p. basis (Eqs. 5–8), Lipkin model (Eqs. 27–30).
- **Gong & Tohyama 1990** — explicit expressions for the `B`, `H`, `P` terms (Eqs. 25, 26, 27).

## Layout

| Path | What |
|---|---|
| [TDDM_Lipkin_September_claude.ipynb](TDDM_Lipkin_September_claude.ipynb) | Everything: TDDM RHS, TDHF, exact Dicke evolution, observables, plots |
| `../Entangle_Magic/Epower_funcs.py` | `Epower_rho(rho, P)` — entanglement from 1-body purity, expects `rho` shape `(2P, 2P, t)` |
| `../Entangle_Magic/fourier_funcs.py` | `fourier(t_list, signal, xlim, ylim, numberpeaks)` |

The imported modules live in a **sibling directory**, not here. Jupyter's cwd is this folder, so
`import Epower_funcs` only resolves if the kernel was started from `Entangle_Magic/` or `sys.path`
was extended. If an import fails, that's why — don't "fix" it by rewriting the module.

Python: `C:\Users\annas\anaconda3\python.exe` (numpy 1.26.4, scipy 1.17.1, opt_einsum 3.4.0).
Bare `python` is not on PATH; use the anaconda path.

## Notation and index conventions

**This is the part to get right.** Everything below is fixed across the notebook.

### Single-particle basis
`P` = number of particles = degeneracy of each level. `2P` s.p. states total.

```
index 0 .. P-1     lower level, energy -eps/2    lo(j) = j
index P .. 2P-1    upper level, energy +eps/2    up(j) = P + j
```

### Greek → einsum letters
Paper indices `α β α' β'` map to einsum `a b c d`, and dummy indices `λ1 λ2 λ3 λ4` to `i j k l`:

```
n[a, c]        = n_{α α'}       = <a+_{α'} a_α>          shape (2P, 2P)
c2[a, b, c, d] = C_{α β α' β'}                            shape (2P,)*4
c3[...]        = C_{λ1λ2λ3, λ1'λ2'λ3'}                    shape (2P,)*6
v[a, b, c, d]  = <α β| v |α' β'>       (bare)
vbar           = v - v.transpose(0,1,3,2)  = <α β|v|α' β'>_A
```

`c2` is antisymmetric in the bra pair `(a,b)` and in the ket pair `(c,d)` — the notebook checks
this every run and it must stay ~0.

### Hamiltonian
`H = eps*Jz + (V/2)(J+² + J-²)`, with `chi = |V|(P-1)/eps`. `chi > 1` is the deformed regime where
the HF ground state breaks symmetry into two minima at `cos(2α) = 1/chi`.

`Hv()` builds the bare `v` in Tohyama's convention, i.e. `H_int = (1/2) Σ v a+a+aa`:
```python
v[up(j), up(k), lo(j), lo(k)] = V   # (V/2) J_+^2
v[lo(j), lo(k), up(j), up(k)] = V   # (V/2) J_-^2
```

## Equations as implemented

`TDDM(t, y)` packs the state as `y = concat(n.ravel(), c2.ravel())`, length `4P² + 16P⁴`,
and multiplies the whole RHS by `-1j` at the very end (the equations are written as `i ṅ = ...`).

**Eq. (7), s.p. energy including mean field** — `ematrix()`:
```
ε_{αα'} = <α|t|α'> + Σ <α λ1|v|α' λ2>_A n_{λ2 λ1}
        = h0 + _es('aibj,ji->ab', vbar, n)
```

**Eq. (5), one-body** — commutator plus the `C2` coupling:
```
d_n = en @ n - n @ en
    + _es('aijk,jkbi->ab', v, c2) - _es('aijk,jkbi->ab', c2, v)
```
The `C2` coupling takes the **bare `v`**, exactly as Eq. (5) is written — *not* `vbar`. `C2` is
antisymmetric in `(λ2, λ3)`, so `vbar` here silently doubles the term and breaks energy
conservation. See "Settled questions" below; this was a real bug, fixed 2026-09-02.

**Eq. (6), two-body** — four `ε·C2` terms, then `B + P + H (+ T)`:
```
d_c2 = εC + Cε terms  +  Bterm + Pterm + Hterm  [+ Tterm]
```
- `Bterm` (Gong Eq. 25): occupation matrices only; 2p–2h and 2h–2p excitations. Uses `vbar` and
  `nb = I - n`.
- `Pterm` (Gong Eq. 27): p–p and h–h correlations to infinite order. Six terms; the two pure
  `M⁶` contractions are reshaped to `(4P², 4P²)` and done as **matmuls** so BLAS handles them.
- `Hterm` (Gong Eq. 26): p–h correlations to infinite order. Sixteen terms, grouped in the code by
  which Kronecker delta they came from (`δ_{αλ1}`, `δ_{βλ2}`, `-δ_{λ3α'}`, `-δ_{λ4β'}`).
- `Tterm` (Tohyama Eq. 8): coupling to `C3`. Currently **inactive** — `build_c3` returns `None` for
  `TRUNCATION = 'TDDM'`, so no `M⁶` array is ever allocated. `'TDDM1'`/`'TDDM2'` raise
  `NotImplementedError`; those are the next thing to build.

## Performance conventions

Two caches exist because the RHS is called thousands of times per integration:

- `statics()` returns `(h0, v, vbar, I)` memoised on `(P, eps, V)`. **It keys off the globals**, so
  changing `P`, `eps` or `V` gives a fresh entry automatically — but changing `Hv()` itself does
  *not* invalidate it. Clear `_STATIC` after editing the Hamiltonian.
- `_es(subs, *ops)` wraps `opt_einsum.contract_expression`, compiled once per
  `(subscripts, shapes)` and cached in `_EXPR`. Use `_es` instead of `np.einsum` inside anything
  that runs per-RHS-call. Plain `np.einsum` is fine for one-off post-processing of solutions.

Integration is `solve_ivp(..., method='RK45', rtol=1e-8, atol=1e-10)`. Timings recorded in the
notebook: `P=3`, 501 points over `t∈[0,5]` → under 20 s; `P=5` → 78 s, nfev 5870.

## Post-processing shapes

Solution arrays carry time as the **last** axis:
```python
rho_soln  = soln.y[:4*P**2].reshape(2*P, 2*P, len(soln.t))
c_soln    = soln.y[4*P**2:].reshape(4*P**2, 4*P**2, len(soln.t))
rho2_soln = A(nn) + c2        # build_rho2(): the antisymmetrisation is REQUIRED, verified
```

Observables all assume that trailing time axis: `Ndown`, `Nup`, `HFstat(rho, th, phi)`,
`Jobservables(n)` → `(<Jx>, <Jy>, <Jz>)`, `lower_occupation(n, c2, N)` → `(mean, var, residual)`.
`Jx/(0.5*P*sin(2*a_dhf))` tracks which DHF minimum the state is nearer: `+1` = min 1, `-1` = min 2.

## Validation checks — run these after touching the RHS

The notebook already has cells for each; they are the whole safety net, since there are no tests.

1. `c2` antisymmetry in bra and ket, over all time (should be ~0).
2. Energy conservation: `Ttot = <h0 ρ>`, `Vtot = 0.5 <ρ2 v>`, `Etot` flat.
3. Particle number `Tr ρ1 = P` conserved.
4. Hermiticity of `ρ1` and `ρ2`.
5. Trace relation `Σ_b ρ2[a,b,a',b] = (N-1) n[a,a']` — the `residual` from `lower_occupation`.
6. For TDHF only: idempotency `ρ² = ρ`.
7. Against exact: `LipkinDicke` evolves in the `(P+1)`-dim Dicke basis; the 3-particle full-space
   `expm` evolution in the "Lipkin exact evolution" section is the other cross-check.

Plot convention when comparing the three methods:
`Exact = mediumvioletred`, `TDDM = darkorange`, `Mean-field = 'b'`.

## Known rough edges — do not silently "fix" these

These are real, currently-open issues in the notebook. Flag them, ask, or fix deliberately — but
don't quietly change a convention that other cells depend on.

- **`chi` gets redefined with a sign.** Cell 1 uses `chi = |V|(P-1)/eps > 0`; the exact-evolution
  cell sets `chi = (P-1)*V/eps`, which is *negative* for attractive `V`. `n_DHF` uses
  `arccos(1/chi)` with positive `chi` — **this one is correct**; the surviving `HFstat` calls in
  the exact-comparison cell still pass `arccos(eps/(V*P-V))`, the signed version, which is
  **wrong** (see "Settled questions"). `a_dhf` and the `Jx/(0.5*P*sin(2*a_dhf))` "which minimum"
  observable use the correct positive-`chi` angle.
- **Entanglement normalisation** is divided by 3 by hand in the comparison cell, marked
  `FIX THE NORMALISATION`. Currently moot: `import Epower_funcs` / `import fourier_funcs` are
  commented out in cell 0 (the sibling-directory problem), so the `Epower_rho` calls are commented
  out too. Re-enabling the imports means re-enabling those cells.

### Fixed as of 2026-09-02 — no longer issues
Previously listed here; corrected in the notebook, recorded so they don't get re-flagged.

- `TDHF` is now defined and live: `d_n = en @ n - n @ en` built from `ematrix`/`statics`, i.e. the
  same mean field as TDDM with `C2` dropped. The old kron-based version is kept as `TDHFold`.
- The TDHF energy cell now reshapes, `Hv().reshape((2*P)**2, (2*P)**2)`, so it matches the live
  `(2P,)⁴` `Hv()`.
- Cell 1 now sets `P = 3` *above* `eps`/`chi`/`V`, so it runs top-to-bottom in a fresh kernel.
- The mid-notebook `P = 4` is commented out.
- `Hdicke`/`vdicke` use `+V/2`, matching `Hv()` — see "Settled questions".
- `observables(n)` was renamed **`Jobservables(n)`** → `(<Jx>, <Jy>, <Jz>)`.

## Settled questions — don't relitigate these

Checked 2026-09-02 against an independent exact many-body calculation (`P=3`, full `2^6` Fock
space, `H` built from `Hv()` and verified equal to `eps*Jz + V/2 (J+² + J-²)`). Method: for a
Slater initial state `C2 = C3 = 0`, so the TDDM RHS must reproduce the exact `dn/dt` and `dC2/dt`
at `t=0`; and TDDM's short-time error against exact must be `O(t⁴)` (missing `C3` only).

- **The `1/2` in `Vtot` is correct.** `Etot = <h0 ρ1> + 0.5 <ρ2 v>` reproduces `E_exact` exactly
  for both initial states. It only *looked* wrong because starting all-in-bottom gives `V(0) = 0`,
  so `0.5·V` and `1.0·V` agree at `t=0` and the check can't discriminate.
- **`Hv()`'s `+V` and its index placement are correct.** It reproduces the Lipkin `H` exactly.
- **`Hdicke`/`vdicke` needed `+V/2`, not `-V/2`** — fixed 2026-09-02. Projecting the exact
  Fock-space `H` (built from `Hv()`) onto the Dicke basis gives `+V/2` on the off-diagonals.
  This was **invisible from the old `|k=0>` start**: flipping that sign is the diagonal gauge
  transformation `U = diag(i^k)`, which leaves the spectrum and every `k`-diagonal observable
  unchanged when you start from `|k=0>`. It is *not* invisible from the DHF minimum, which has
  weight on every `k` — `|<0|ψ(t)>|²` at `t = 1, 3, 5` went `0.221, 0.131, 0.181` (wrong sign)
  vs `0.275, 0.393, 0.326` (correct). Any future "the Dicke result disagrees with TDDM" should
  check this first.
- **The DHF minimum in the Dicke basis** is `<k|DHF> = sqrt(C(P,k)) cos(a)^(P-k) sin(a)^k` with
  `cos(2a) = 1/chi`, `chi > 0` — `psi_DHF_dicke()`. It is an SU(2) coherent state, so it lies
  entirely inside the `(P+1)`-dim symmetric subspace (verified: zero weight outside, overlaps
  match the full `2^(2P)` Fock-space Slater to 1e-16) and is normalised without rescaling.
- **Eq. (5) takes bare `v`, not `vbar`** — this was the actual bug behind the energy-conservation
  failure. Evidence:

  | one-body term | short-time error vs exact | `E=T+0.5V` drift, HF min | `E=T+0.5V` drift, bottom |
  |---|---|---|---|
  | `vbar` (old) | `O(t²)` | 2.6 | 1.4 |
  | `v` (fixed)  | `O(t⁴)` | 3e-10 | 2e-15 |

  With `vbar`, starting all-in-bottom, `T + 1.0·V` is conserved to 3e-15 — a **coincidence of that
  one initial state**, and the reason the missing-`1/2` story looked plausible. It does not hold
  from the HF minimum, where nothing was conserved.
- **`n_DHF`'s HF angle is the correct one.** With `cos(2α) = 1/χ`, `χ = |V|(N-1)/ε > 0`, the
  Slater energy is `-3.1875`, matching Tohyama Eq. (20) `Nε/4·(-χ - 1/χ)` exactly, and a scan over
  `θ` finds the minimum there. `HFstat`'s `arccos(eps/(V*P-V))` uses signed `χ`, giving `-2.4375`
  — not a stationary point. Fix the `HFstat` call sites, not `n_DHF`. This resolves the notebook's
  "NOT SURE WHICH HF MIN FUNCTION IS CORRECT".

## Working style

- Physics conventions are load-bearing. When changing an index order, a sign, or an
  antisymmetrisation, say which paper equation it comes from and re-run the validation checks
  above rather than eyeballing a plot.
- Keep the paper's index letters (`a b c d` = `α β α' β'`) in every new einsum. Deviating makes the
  transcription impossible to check against Gong Eqs. 25–27.
- The commented-out code in the notebook is deliberate history (previous formulations that were
  tested and superseded). Leave it unless asked. In particular, ignore the TDHF cells because they are old and wont work and arent relevant right now
- Figures save to `../Entangle_Magic/` with `bbox_inches='tight', dpi=300`; those calls are
  commented out by default.
