# BPO: Bayesian Preference Optimization for Multiobjective Discrete Optimization

This repository implements Bayesian Preference Optimization (BPO) for multiobjective discrete optimization, with a concrete instantiation on the Multi-Objective Knapsack Problem (MOKP). BPO learns where to evaluate scalarization preferences on the probability simplex and selects the next preference weights by maximizing a multiobjective acquisition function.

At each BO iteration, we:
- Sample or optimize preference weights `lambda` on the simplex (sum to 1, nonnegative).
- Solve the discrete subproblem via an augmented Tchebycheff scalarization using Gurobi to obtain an objective vector.
- Fit/update a GP surrogate mapping `lambda -> objective vector`.
- Select new `lambda` with a hypervolume-based acquisition (qLogEHVI) or a baseline random Dirichlet sampler.

The result is an efficient, model-based exploration of the Pareto frontier for discrete problems.

## Key Features

- Preference-space BO over the simplex with equality constraints (sum of weights = 1).
- Multiobjective acquisition: qLogExpectedHypervolumeImprovement (qLogEHVI) from BoTorch.
- Exact discrete solves for each `lambda` using Gurobi on an augmented Tchebycheff scalarization.
- Modular problem interface to plug in new multiobjective discrete problems.
- Reproducible runs with clear output artifacts (JSON) including nondominated sets and iteration logs.

## Repository Structure

- `src/run_bpo.py`: CLI entry point to run preference-based BO end-to-end.
- `src/acquisition.py`: Acquisition registry and implementations (qLogEHVI and random Dirichlet).
- `src/bpo/core/`
  - `config.py`: BO configuration container (seeds, budgets, acquisition settings).
  - `run.py`: Main BO loop, model fitting, candidate generation, HV tracking, and result saving.
  - `model.py`: Multi-output GP surrogate (ModelListGP over standardized single-task GPs).
  - `io.py`: Output structure and JSON writer for runs.
- `src/bpo/problems/`
  - `base.py`: Abstract `Problem` interface (bounds, constraints, evaluation, IO metadata).
  - `mokp.py`: MOKP implementation with augmented Tchebycheff scalarization solved by Gurobi.
  - `__init__.py`: Problem registry and builder.

## MOKP + Augmented Tchebycheff

Given preference weights `lambda` on the simplex, we solve a single mixed-integer problem using an augmented Tchebycheff objective that balances the max-deviation from the ideal point with a small augmentation term (`rho`) to avoid weakly Pareto-optimal solutions. The solver returns the true multiobjective vector (maximization convention configurable), which serves as supervised data for the surrogate.

Notes:
- Initial design points are drawn from a Dirichlet over the simplex.
- Equality constraint `sum(lambda) = 1` is enforced during acquisition optimization.
- Hypervolume is optionally normalized by the product of the absolute ideal point components when available.

## Installation

Prerequisites:
- Python 3.9+
- A working Gurobi installation and license (for `gurobipy`).
- PyTorch compatible with your platform/GPU.

Steps:
- Install PyTorch following the official instructions for your platform.
- Install the remaining dependencies:
  - `pip install -r requirements.txt`

The BoTorch/GPyTorch dependencies are pulled in via `botorch`, but PyTorch must typically be installed first.

## Quick Start

Basic run on MOKP (3 objectives, 50 items):

```
python src/run_bpo.py \
  --problem mokp \
  --acquisition qlogehvi \
  --n-items 50 --n-objs 3 --density 0.5 \
  --n-initial-samples 10 --n-iterations 20 \
  --batch-size-q 2 --mc-samples 128 --raw-samples 512 \
  --rseed 123 --iseed 123 --rho 1e-4 \
  --should-maximize true --sequential true
```

Switch to a random baseline acquisition:

```
python src/run_bpo.py --acquisition random
```

Reference point for HV (optional, one value per objective):

```
python src/run_bpo.py --ref-point 0 0 0
```

## Outputs

Run artifacts are stored under `outputs/…` with a problem- and acquisition-specific directory chain. Each run writes a timestamped JSON file containing:
- Problem metadata and config
- Acquisition settings
- Per-iteration metrics (hypervolume, nondominated count)
- Final nondominated set of objective vectors

Example directory pattern for MOKP:
- `outputs/mokp-items-<items>_objs-<objs>_iseed-<iseed>_rseed-<rseed>/<acq>/n_initial_samples-<n>/…`

## Extending

To add a new problem:
- Subclass `bpo.problems.base.Problem` and implement:
  - `n_objectives()`, `lambda_bounds()`, and optionally `lambda_equality_constraints()`
  - `initial_design(n)` and `evaluate(lambda_batch, maximize=True)`
  - `metadata()` and `io_base_dir(config)`
- Register it in `bpo/problems/__init__.py` so it appears in `--problem` choices.

To add a new acquisition:
- Implement a subclass of `AcquisitionFunction` in `src/acquisition.py`.
- Register it in `ACQUISITION_REGISTRY` so it appears in `--acquisition` choices.

## Reproducibility

- `--rseed` controls Torch, BoTorch sampler seeds, and NumPy/Python RNG used in the loop.
- `--iseed` controls the problem instance (e.g., MOKP profits/weights and capacity).

## License

This project is licensed under the terms of the MIT License. See `LICENSE`.
