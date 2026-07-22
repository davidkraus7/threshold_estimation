# Threshold Estimation in Endogenously Switching SVARs

MPhil thesis project. Estimates the regime threshold
`gamma_0` in censored/kinked and endogenous regime switching SVARs, where the
existing literature (Mavroeidis 2021; Duffy–Mavroeidis 2026) takes it as given.
The literature and write-ups live alongside this repository in
`../01. literature` and `../02. docs`.

The central result is that `gamma_0` is *super-consistently* estimable — at rate
`T` rather than the usual `sqrt(T)` — even though the model's conditional mean is
continuous, a kink rather than a jump. The mechanism is simultaneity: the
likelihood's Jacobian is regime-dependent, so the density of the threshold
variable jumps at `gamma_0` by the factor `x = (1 - delta*kappa_2)/(1 - delta*kappa_1)`.
When the system is simultaneous (`x != 1`) that density break pins the threshold
down at rate `T`; when it is recursive (`delta = 0`, so `x = 1`) the break
vanishes and the rate falls to `sqrt(T)`. Simultaneity is the source of
identification, not an obstacle to it.

## Project Structure

```
├── simulations/       Monte Carlo suite — active workstream (see simulations/README.md)
├── environment.yml    conda environment `thresh`
├── pyproject.toml     package and pytest configuration
├── CLAUDE.md          model, notation, and working rules
└── README.md          this file
```

Later workstreams — estimation for richer model classes, and the empirical
application — get their own sibling directories. Code stays inside the
workstream that uses it until something is shared, at which point it moves up to
the repository root.

## Quick Start

1. **Install** Miniconda or Anaconda.
2. **Create the environment**: `conda env create -f environment.yml`.
3. **Activate it**: `conda activate thresh`.
4. **Run the tests**: `pytest`.
5. **Run an experiment** from this directory (the repository root):
   `python simulations/experiments/e2_rate.py`.
6. **View outputs** in `simulations/results/` (PDF figures, CSV tables).

## Dependencies

Managed via `conda` (`environment.yml`, environment name `thresh`, Python 3.12).
Key packages:

- `numpy`, `scipy` -- numerics and optimisation
- `pandas`, `pyarrow` -- data handling and parquet caches
- `matplotlib` -- figures
- `sympy` -- symbolic derivations
- `pytest` -- tests

## Conventions

- Results are never tracked in git; each results file carries the commit hash
  (`git_sha`) that produced it, so any figure traces back to its code.
- Commit before any run you intend to keep — an uncommitted tree stamps results
  `-dirty`.
- Tag the repository when exhibits go to the supervisor:
  `git tag -a exhibits-2026-07-24 -m "..."`.
