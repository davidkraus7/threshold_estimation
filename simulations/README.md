# Simulations

Monte Carlo evidence for the central claim of the thesis: `gamma_0` is estimable
at rate `T` when the threshold variable's density jumps at `gamma_0` (the
simultaneous design, `x != 1`), and only at rate `sqrt(T)` when it does not (the
recursive design, `x = 1`). The full experimental design is specified in
`../../02. docs/03. simulation plan/simulation_plan.tex`; this directory
implements it.

Two designs share a baseline (`gamma_0 = 0`, `c_2 = 0.3`, `kappa_1 = 0.25`,
`kappa_2 = 1.5`, `sigma = 1`) and differ only in the feedback `delta`: the
**endogenous** design (`delta = 0.4`, `x = 0.444`) is simultaneous, the
**exogenous** design (`delta = 0`, `x = 1`) is recursive. Common random numbers
across designs isolate the effect of `delta` alone.

## Project Structure

```
threshold_estimation/   the shared library
├── config.py           every parameter, the seed policy, paths
├── dgp.py              coherency-checked branch solver
├── likelihood.py       concentrated Gaussian log-likelihood incl. the Jacobian
├── estimation.py       grid search over gamma with multi-basin refinement
└── analytics.py        closed-form population quantities (density jump, KL slopes)
experiments/            one script per experiment (E1-E6)
results/                gitignored; regenerable from the scripts
tests/                  one file per failure mode
```

`config.py` holds every parameter of every experiment; nothing else defines a
numerical constant.

## Quick Start

Run from the repository root (`03. code`), with the `thresh` environment active:

1. **E1** (criterion shape): `python simulations/experiments/e1_criterion.py`.
2. **E2** (convergence rate): `python simulations/experiments/e2_rate.py`.
3. **Rebuild a figure** from cached results without re-simulating: add
   `--figures-only` (E2) or `--replot` (E1).
4. **View outputs** in `results/` (PDF figures, CSV tables, parquet caches).

For long runs, prefix with `caffeinate -i` and leave the lid open.

## Experiments

| Script | Exhibit | Status |
|---|---|---|
| `e1_criterion.py`   | Figure 1: shape of the profile criterion near `gamma_0` | done |
| `e2_rate.py`        | Figure 2, Table 1: RMSE of `gamma_hat` against `T`      | done |
| `e3_limit.py`       | Figure 3: shape of the limit distribution              | stub |
| `e4_as_if_known.py` | Table 2: coverage treating `gamma_hat` as known        | stub |
| `e5_endogeneity.py` | Figure 4: RMSE against the endogeneity dial `delta`    | stub |
| `e6_thin_regimes.py`| Small table: thin regimes and weak kinks               | stub |

## Estimator

Gaussian MLE. For each candidate `gamma` on a grid of order statistics of `q`
(trimmed to the central 80%, thinned to 96 points), `(c1, c2, sigma1, sigma2)`
are concentrated out in closed form and `(kappa1, kappa2, delta)` are optimised
by Nelder–Mead. The coarse grid is refined around its best 8 candidates so the
peak is not missed, and `gamma_hat` is the maximising order statistic.

## Seeds

`config.replication_rng(experiment, rep)` is keyed on the experiment name and
replication index only, never on design parameters — that is what makes the two
designs common-random-number comparable. Draw with `config.standard_shocks` and
scale by `sigma` in the DGP. Shocks are nested across `T` (the `T = 256` sample
is a prefix of the `T = 8192` sample), which correlates replications across
sample sizes and reduces the noise in the rate estimate.

## Results and Tests

Each experiment writes one parquet into `results/`, stamped with `git_sha` and
the estimator label. `tests/test_config.py` guards the seed policy and design
coherency; the remaining test files correspond to the failure modes flagged in
`../../02. docs/04. roadmap/roadmap.tex` — a missing Jacobian term, a continuity
restriction not tracking `gamma`, and coherency violations passing silently.

Dependencies are managed at the repository root; one `thresh` environment serves
the whole project. See `../README.md`.
