# Bayesian Preference Optimization

This repository is the official implementation of the manuscript: Bayesian Preference Optimization for Multiobjective Discrete Optimization. Alongside the Bayesian optimization pipeline for preference-guided search, it offers companion baselines based on random preference selection and evolutionary algorithms. 
It provides a unified workflow for defining multiobjective problems, building scalarized views of those problems, and evaluating a catalog of solvers.

## Architecture

```
src/
  problem/        # Problem generators and metadata
  scalarization/  # Scalarized reformulations (e.g., augmented Chebyshev)
  solver/         # Solver implementations (BPO, randomized, evolutionary)
  configs/        # Hydra configuration hierarchy
  utils/          # Shared helpers (seeding, sampling, paths, etc.)
run_*.py          # Entry points for each solver
```

- **problem** – creates multiobjective instances (e.g., random landscapes, combinatorial problems) and exposes common metadata such as dimensionality, capacities, and cached ideal points.
- **scalarization** – houses reusable scalarizers that convert multiobjective queries into single-objective subproblems. Solvers that rely on scalarization (BPO, random sampling) compose these modules with problem instances.
- **solver** – contains solver-specific orchestration:
  - `solver/bpo` runs Bayesian preference optimization using BoTorch-based surrogates and acquisition functions.
  - `solver/aug_cheby` samples random preference vectors (Dirichlet) and evaluates them via a scalarizer.
  - `solver/ea` provides evolutionary algorithm baselines implemented directly against `pymoo`.
  Each solver manages its own logging, hypervolume tracking, and JSON output.
- **configs** – Hydra configuration tree for problems, scalarizations, solvers, and experiments.
- **utils** – convenience functions for seeding, Dirichlet sampling, normalization, and output directories.

Top-level `run_*.py` scripts wire the pieces together. They load Hydra configs, instantiate problem instances, hand them to solver classes, and emit results under `outputs/...`.

## Running Solvers

Hydra manages configuration. Override any parameter with dot-notation at the CLI. The examples below illustrate common patterns—replace placeholders (`<...>`) with values defined in your config tree.

### Bayesian Preference Optimization (BPO)

```
python src/run_bpo.py \
  problem.name=<problem_id> \
  scalarization.rho=1e-4 \
  bo.n_iterations=50 \
  bo.time_limit=300 \
  acquisition.name=qlogehvi
```

This command launches the preference-based BO loop using the augmented Chebyshev scalarizer. You can swap surrogate or acquisition settings via `configs/surrogate/*.yaml` and `configs/acquisition/*.yaml` (see the Hydra defaults inside `src/configs/run_bpo.yaml`).

### Random Augmented Chebyshev Sampling

```
python src/run_aug_cheby.py \
  problem.name=<problem_id> \
  scalarization.rho=1e-4 \
  n_iterations=200 \
  time_limit=120 \
  rseed=42
```

The solver samples one preference vector per iteration from a Dirichlet, evaluates it through the scalarizer, and records the evolving Pareto front.

### Evolutionary Algorithms

```
python src/run_ea.py \
  problem.name=<problem_id> \
  algorithm.name=nsga2 \
  algorithm.pop_size=200 \
  algorithm.time=00:05:00 \
  algorithm.seed=123
```

Available algorithms include `nsga2`, `nsga3`, `smsemoa`, and `ctaea`. Each algorithm’s parameters live under `configs/algorithm/<name>.yaml`. Override or extend them via Hydra (e.g., `algorithm.n_partitions=12`).

## Configuration & Customization

- **Problems** – add new instances under `src/problem/` and register their configs in `src/configs/problem/`. Problems should expose metadata, a reference point, and a cached ideal point.
- **Scalarizations** – implement new formulations in `src/scalarization/`; solvers can compose them with problem instances for preference evaluation or other transformations.
- **Solvers** – inherit from existing solver patterns or create new ones under `src/solver/`. Each solver is responsible for its execution loop, metrics, and output serialization.
- **Hydra Defaults** – the files in `src/configs/run_*.yaml` declare default config chains. Modify these defaults or supply CLI overrides to experiment with different setups.

## Outputs

Every run produces a timestamped JSON report in `outputs/<solver>/...`. Directory chains encode the problem descriptor (via `str(instance)`), random seeds, and solver-specific settings.

Examples:

- `outputs/bpo/<problem_descriptor>/surr-<name>/acq-<name>/n_init-<N>/n_iter-<N>/.../run_bo_<timestamp>.json`
- `outputs/aug_cheby/<problem_descriptor>/rseed-<seed>/n_iter-<N>/time-<seconds>/run_aug_cheby_<timestamp>.json`
- `outputs/ea/<problem_descriptor>_seed-<seed>/algorithm-<name>/pop_size-<N>/time-<hh-mm-ss>/run_ea_<timestamp>.json`

Files contain the resolved Hydra config, problem metadata, per-iteration metrics (hypervolume, nondominated counts), nondominated sets, and run timing.

## Extending the Playground

1. **Define a Problem** – create a generator in `problem/` that produces objective vectors/constraints and caches the ideal point. Add a Hydra config under `configs/problem/` to expose it.
2. **Provide Scalarizations (optional)** – if the solver requires a scalarized subproblem, implement it in `scalarization/` and compose it with the problem instance.
3. **Build a Solver** – follow the patterns in `solver/bpo`, `solver/aug_cheby`, or `solver/ea` to implement training loops, candidate generation, and result serialization.
4. **Register Configs** – extend the corresponding `run_*.yaml` or create new entry points for custom experiments.

## Requirements & Setup

- Python 3.8+
- Gurobi (for scalarizers that solve mixed-integer programs) with a valid license
- PyTorch and BoTorch (installed via `requirements.txt`)
- Optional: CUDA for GPU acceleration, `pymoo` for evolutionary algorithms

Install dependencies:

```
pip install -r requirements.txt
```

The code runs in float64 by default for numerical stability.

## Reproducibility

- `problem.iseed` controls the stochastic generation of problem instances.
- `bo.rseed`, `n_iterations`, and related fields govern the BO loop’s randomness.
- `rseed` in `run_aug_cheby` sets Dirichlet sampling seeds.
- `algorithm.seed` (and `algorithm.time`) configure evolutionary runs.

All solvers record their resolved configs alongside results to aid reproducibility.

## License

Distributed under the MIT License. See `LICENSE` for details.
