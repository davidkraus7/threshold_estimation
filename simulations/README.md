# Simulations

Monte Carlo evidence for the central claim of the thesis: that `gamma_0` is
estimable at rate `T` when the determinant ratio `x != 1`, and only at rate
`sqrt(T)` when `x == 1`. The experimental design is specified in
`../../02. docs/03. simulation plan/simulation_plan.tex`; this directory
implements it.

## Layout

```
threshold_estimation/   config, dgp, likelihood, estimation, analytics
experiments/            one script per experiment, E1-E6
results/                gitignored; regenerable from the scripts
tests/                  one file per bug class flagged in the roadmap
```

`threshold_estimation/config.py` holds every parameter of every experiment;
nothing else defines a numerical constant.

## Experiments

| Script | Exhibit |
|---|---|
| `e1_criterion.py` | Figure 1: shape of the profile criterion near `gamma_0` |
| `e2_rate.py` | Figure 2, Table 1: RMSE of `gamma_hat` against `T` — the headline |
| `e3_limit.py` | Figure 3: shape of the limit distribution |
| `e4_as_if_known.py` | Table 2: coverage treating `gamma_hat` as known |
| `e5_endogeneity.py` | Figure 4: RMSE against the endogeneity dial `delta` |
| `e6_thin_regimes.py` | Small table: thin regimes and weak kinks |

Run from the repository root: `python simulations/experiments/e2_rate.py`.

## Seeds

`config.replication_rng(experiment, rep)` is keyed on the experiment name and
replication index only, never on design parameters. That is what makes the
endogenous and exogenous columns of each figure common-random-number
comparable. Draw with `config.standard_shocks` and scale by sigma in the DGP;
do not consume the generator for anything else first.

`config.standard_shocks` returns the full `T_MAX` path by default, so the
`T = 250` sample is a prefix of the `T = 4000` sample. That nesting correlates
replications across sample sizes and reduces the noise in the E2 rate estimate.

## Results

Each experiment writes one parquet into `results/`, stamped with the commit
that produced it:

```python
df["git_sha"] = config.git_sha()
df.to_parquet(config.RESULTS / "e2_rate.parquet")
```

## Tests

`tests/test_config.py` runs today and guards the seed policy. The other three
files are placeholders naming what to check, one per failure mode flagged in
`../../02. docs/04. roadmap/roadmap.tex`: a missing Jacobian term, a continuity
restriction not tracking `gamma`, and coherency violations passing silently.
Fill them in as the modules they cover get written.
