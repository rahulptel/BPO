# Bayesian Preference Optimization

This repository is the official implementation of the manuscript: Bayesian Preference Optimization for Multiobjective Discrete Optimization. It provides a unified workflow for generating multiobjective discrete optimization instances, scalarizing them with preference vectors, and comparing Bayesian preference optimization against exact and heuristic baselines.

## Architecture

```
src/
  problem/        # MOKP and MOAP instance generators and metadata
  scalarization/  # Scalarized reformulations (e.g., augmented Chebyshev)
  solver/         # Solver implementations (BPO, random scalarization, EA, KSA)
  configs/        # Hydra configuration hierarchy
  utils/          # Shared helpers for seeding, Gurobi, hypervolume, and paths
  run_*.py        # Entry points for each solver
```

- **problem** – creates random multiobjective knapsack (`mokp`) and assignment (`moap`) instances. Each instance exposes metadata, an ideal point, and a reference point for hypervolume computation.
- **scalarization** – houses augmented Chebyshev scalarizers for MOKP and MOAP. Gurobi and SCIP backends are available for scalarized subproblem solves.
- **solver** – contains solver-specific orchestration:
  - `solver/bpo` runs Bayesian preference optimization with a BoTorch GP surrogate and qLogEHVI acquisition over the preference simplex.
  - `solver/aug_cheby` samples Dirichlet preference vectors and evaluates each vector with the augmented Chebyshev scalarizer.
  - `solver/ea` runs `pymoo` baselines with binary encoding for MOKP and permutation encoding for MOAP.
  - `solver/ksa` runs an epsilon-constraint style Kirlik-Sayin search over exact Gurobi subproblems.
- **configs** – Hydra configuration tree for problems, scalarizations, solvers, and experiments.
- **utils** – shared functions for seeding, Gurobi environment setup, nondominated iteration records, and hypervolume computation.

Top-level `run_*.py` scripts wire the pieces together. They load Hydra configs, instantiate problem instances, hand them to solver classes, and emit results under `outputs/...`.

## Running Solvers

Hydra manages configuration. Override any parameter with dot-notation at the CLI. The examples below illustrate common patterns using the current config tree.

### Bayesian Preference Optimization (BPO)

```
python src/run_bpo.py \
  problem=mokp \
  scalarization.rho=1e-4 \
  n_iterations=50 \
  time_limit=300 \
  acquisition.name=qlogehvi
```

This command launches the preference-based BO loop using the augmented Chebyshev scalarizer. The default surrogate is `configs/surrogate/gp.yaml`; acquisition settings are defined inline in `src/configs/run_bpo.yaml`.

### Random Augmented Chebyshev Sampling

```
python src/run_aug_cheby.py \
  problem=moap \
  scalarization.rho=1e-4 \
  n_iterations=200 \
  time_limit=120 \
  rseed=42
```

The solver samples one preference vector per iteration from a Dirichlet, evaluates it through the scalarizer, and records the evolving Pareto front.

### Evolutionary Algorithms

```
python src/run_ea.py \
  problem=mokp \
  algorithm=nsga2 \
  algorithm.pop_size=200 \
  algorithm.time=00:05:00 \
  algorithm.seed=123
```

Available algorithms include `nsga2`, `nsga3`, `smsemoa`, and `ctaea`. Each algorithm’s parameters live under `configs/algorithm/<name>.yaml`. Override or extend them via Hydra (e.g., `algorithm.n_partitions=12`).

### Kirlik-Sayin Approximation (KSA)

```
python src/run_ksa.py \
  problem=moap \
  objective_index=0 \
  delta=1 \
  time_limit=300 \
  mem_limit_gb=16
```

KSA uses Gurobi-backed epsilon-constraint subproblems and currently requires `optimizer=gurobi`. Set `save_solutions=true` to include decision vectors in the JSON output.

## Configuration & Customization

- **Problems** – add new instances under `src/problem/` and register their configs in `src/configs/problem/`. Problems should expose `name`, `metadata()`, `reference_point`, `ideal_point`, and a stable `__str__` descriptor for output paths.
- **Scalarizations** – implement new formulations in `src/scalarization/` and register them in `build_scalarizer`. BPO and random augmented Chebyshev sampling expect scalarizers with an `evaluate(prefs)` method and `n_evaluations` counter.
- **Solvers** – follow the existing solver patterns under `src/solver/`. Each solver owns its execution loop, metrics, and JSON output serialization.
- **Hydra Defaults** – the files in `src/configs/run_*.yaml` declare default config chains. Modify these defaults or supply CLI overrides to experiment with different setups.

## Outputs

Every run produces a timestamped JSON report in `outputs/<solver>/...`. Directory chains encode the problem descriptor (via `str(instance)`), random seeds, and solver-specific settings.

Examples:

- `outputs/bpo/<problem_descriptor>/surr-<name>/acq-<name>/n_init-<N>/n_iter-<N>/.../run_bo_<timestamp>.json`
- `outputs/aug_cheby/<problem_descriptor>/rseed-<seed>/n_iter-<N>/time-<seconds>/run_aug_cheby_<timestamp>.json`
- `outputs/ea/<problem_descriptor>_seed-<seed>/algorithm-<name>/pop_size-<N>/time-<hh-mm-ss>/run_ea_<timestamp>.json`
- `outputs/ksa/<problem_descriptor>/time-<seconds>/run_ksa_<timestamp>.json`

Files contain the resolved Hydra config, objective vectors, run timing, and solver-specific metrics. BPO, random augmented Chebyshev, and KSA runs also store per-iteration hypervolume and nondominated counts. EA runs store the final nondominated set and final hypervolume.

## Extending the Playground

1. **Define a Problem** – create a generator in `problem/` that produces objective vectors/constraints and caches the ideal point. Add a Hydra config under `configs/problem/` and update `build_instance`.
2. **Provide Scalarizations (optional)** – if the solver requires a scalarized subproblem, implement it in `scalarization/` and compose it with the problem instance.
3. **Build a Solver** – follow the patterns in `solver/bpo`, `solver/aug_cheby`, `solver/ea`, or `solver/ksa` to implement training loops, candidate generation, and result serialization.
4. **Register Configs** – extend the corresponding `run_*.yaml` or create new entry points for custom experiments.

## Requirements & Setup

- Python 3.8+
- Gurobi with a valid license for the default optimizer and KSA
- SCIP/PySCIPOpt is available as an alternative scalarizer backend for BPO and random augmented Chebyshev runs via `optimizer=scip`
- PyTorch, BoTorch, pymoo, pygmo, Hydra, and related packages from `requirements.txt`

Install dependencies:

```
pip install -r requirements.txt
```

The code runs in float64 by default for numerical stability.

## Reproducibility

- `problem.iseed` controls the stochastic generation of problem instances.
- `from_pid` and `to_pid` run consecutive instance seeds by setting `problem.iseed` for each `pid` in `range(from_pid, to_pid)`.
- `rseed`, `n_initial_samples`, `n_iterations`, and related fields govern the BPO loop’s randomness.
- `rseed` in `run_aug_cheby` sets Dirichlet sampling seeds.
- `algorithm.seed` (and `algorithm.time`) configure evolutionary runs.
- `objective_index`, `delta`, and `time_limit` configure KSA.

All solvers record their resolved configs alongside results to aid reproducibility.

## License

Distributed under the MIT License. See `LICENSE` for details.
