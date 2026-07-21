# MPhil thesis code

Code for the thesis on estimating the regime threshold `gamma_0` in censored
and kinked SVARs, where the literature treats it as known. Documents and
literature live alongside this repository, in `../01. literature` and
`../02. docs`.

## Workstreams

| Directory | Status |
|---|---|
| `simulations/` | Monte Carlo evidence on the rate of convergence of `gamma_hat`. Active. |

Later work — estimation for richer model classes, and the empirical
application — gets its own sibling directory. Anything shared by more than one
workstream moves up to this level; until then, code stays inside the workstream
that uses it.

## Setup

One environment serves the whole repository:

```sh
conda env create -f environment.yml
conda activate thresh
pytest
```

Scripts are run from this directory, the repository root:

```sh
python simulations/experiments/e2_rate.py
```

## Conventions

- Each workstream owns its outputs and ignores them in its own `.gitignore`;
  the root `.gitignore` covers only language and OS artefacts.
- Results are never tracked. They carry the commit hash that produced them
  instead, so a figure can be traced back to its code.
- Tag the repository whenever exhibits go to the supervisor:
  `git tag -a exhibits-2026-07-24 -m "Figures 1-4, Tables 1-2"`.

## Remote

Not configured; `gh` is not installed on this machine. Once a GitHub
repository exists:

```sh
git remote add origin git@github.com:davidkraus7/threshold_estimation.git
git push -u origin main
```
