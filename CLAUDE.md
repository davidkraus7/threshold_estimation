# Project context

MPhil thesis (Oxford econometrics): estimating the regime threshold `gamma_0`
in censored/kinked and endogenously switching SVARs, where the literature
(Mavroeidis 2021 Ecta; Duffy–Mavroeidis 2026) treats it as known. This
repository holds all code for the thesis; `simulations/` is the first
workstream, with richer model classes and the empirical application to follow.

## The claim under test

The piecewise-affine SVAR class is continuous by construction, so there is no
conditional-mean jump at the threshold and Chan's (1993) superconsistency
argument does not apply. But the likelihood's Jacobian term
`log|det Phi_0^(s_t)|` is a step function, so the conditional density of the
threshold variable jumps at `gamma_0` by the determinant ratio `x`. The local
KL expansion `KL(gamma_0 +/- eta) = b_pm * eta + O(eta^2)` then gives the
`T`-rate iff `x != 1`, and the `sqrt(T)`-normal rate (Chan–Tsay 1998, Hansen
2017) iff `x == 1`. Punchline: simultaneity is the source of superconsistency.

Neither Chan (1993) (his Condition 4 fails) nor Hansen (2000) (A1.5 excludes
regime-dependent variance, A1.7 excludes continuous means) covers this case.

## Running example and notation

The simulation workstream uses the smallest model exhibiting endogenous
switching — a static bivariate piecewise-linear SEM, `z_t = (pi_t, q_t)'`:

    pi_t = c1 + kappa1 * min(q_t - gamma0, 0) + kappa2 * max(q_t - gamma0, 0) + eps_1t
    q_t  = c2 + delta * pi_t + eps_2t,        eps_t ~ iid N(0, diag(sigma1^2, sigma2^2))

with `Phi_0^(l) = [[1, -kappa_l], [-delta, 1]]`, so `det Phi_0^(l) = 1 - delta*kappa_l`.

| Symbol | Meaning |
|---|---|
| `gamma0` | true threshold |
| `kappa_1, kappa_2` | slopes below and above the threshold |
| `delta` | endogeneity dial; `delta = 0` is the exogenous benchmark |
| `x` | determinant ratio `(1 - delta*kappa_2)/(1 - delta*kappa_1)` |
| `d(x)` | `x log x - x + 1`, non-negative, zero iff `x = 1` |
| `b_pm` | one-sided slopes of the population criterion, `f_mp(gamma0) * d(x^{+/-1})` |

## Failure modes to hunt

From `../02. docs/04. roadmap/roadmap.tex:324`. All three converge without
error and corrupt the result; they recur wherever this model class is
estimated, not only in the simulations.

1. **Missing Jacobian term** in the likelihood. It is the entire source of
   information about `gamma_0`; drop it and the `T`-rate fails mechanically.
2. **Continuity restriction not tracking gamma.** Kink regressors must be
   rebuilt at every candidate `gamma`. Pin continuity at `gamma_0` instead and
   a mean jump leaks in, making the estimator superconsistent for the wrong
   reason.
3. **Silent coherency violations.** Assert exactly one admissible branch per
   observation rather than trusting the parameter-level check.

## Working rules

- **No lemma without a companion script** (roadmap.tex:325). Every analytical
  claim gets a numerical check; finite-difference every derivative.
- **Attack, never confirm.** When checking a proof, find the weakest step and
  attempt a counterexample. Agreement bias otherwise polishes errors.
- **Never cite a paper that has not been opened.** Quote and give section
  references, working against the files; fabricated or subtly wrong citations
  are the characteristic failure mode.
- Deliverable documents are self-contained Overleaf-ready `.tex` with
  `filecontents` bibs — there is no local LaTeX toolchain on this machine.

## Pointers

- Proposal: `../02. docs/01. proposal/proposal.md`
- Referee-style feedback: `../02. docs/02. feedback/feedback.tex`
- Simulation design: `../02. docs/03. simulation plan/simulation_plan.tex`
- Reading and proof roadmap: `../02. docs/04. roadmap/roadmap.tex`
- Paper library (OCR'd markdown — read these, not the PDFs):
  `/Users/david/Documents/05. misc/01. claude skills.nosync/phd-literature/papers/projects/threshold_estimation/`
