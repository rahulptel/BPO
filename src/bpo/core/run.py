import random
import time

import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated

from acquisition import AcquisitionConfig, build_acquisition
from .model import initialize_model
from .io import save_result


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_acquisition_function(problem, config, ref_point):
    bounds = problem.lambda_bounds()
    equality_constraints = problem.lambda_equality_constraints()
    acquisition_config = AcquisitionConfig(
        ref_point=ref_point,
        bounds=bounds,
        batch_size=config.batch_size_q,
        num_restarts=config.num_restarts,
        raw_samples=config.raw_samples,
        sequential=config.sequential,
        equality_constraints=equality_constraints,
        mc_samples=config.mc_samples,
        rseed=config.rseed,
    )
    return build_acquisition(config.acquisition, acquisition_config)


def _normalize_hypervolume(problem, value):
    ideal_point = problem.ideal_point()
    if ideal_point is None:
        return value
    denom = torch.abs(ideal_point).prod().item()
    if denom == 0:
        return value
    return value / denom


def run_bo(problem, config):
    set_global_seed(config.rseed)

    print(f"Generating {config.n_initial_samples} initial data points...")
    train_lambda = problem.initial_design(config.n_initial_samples)
    train_obj = problem.evaluate(train_lambda, maximize=config.should_maximize)
    print("Initial data generation complete.")

    if config.ref_point is None:
        ref_point = problem.default_ref_point()
    else:
        ref_point = config.ref_point.to(dtype=torch.get_default_dtype())
    config.ref_point = ref_point

    acquisition_function = _build_acquisition_function(problem, config, ref_point)
    print(
        f"Using acquisition function: {acquisition_function.__class__.__name__} | rseed: {config.rseed}"
    )
    print(f"Starting BO loop for {config.n_iterations} iterations...")

    start_time = time.time()
    iteration_records = []

    for iteration in range(config.n_iterations):
        mll, model = initialize_model(train_lambda, train_obj)
        fit_gpytorch_mll(mll)

        new_lambda = acquisition_function.generate_candidates(
            model, train_lambda, train_obj
        )
        new_obj = problem.evaluate(new_lambda, maximize=config.should_maximize)

        train_lambda = torch.cat([train_lambda, new_lambda])
        train_obj = torch.cat([train_obj, new_obj])

        pareto_mask = is_non_dominated(train_obj)
        bd = FastNondominatedPartitioning(ref_point=ref_point, Y=train_obj)
        hypervolume = bd.compute_hypervolume().item()
        hypervolume = _normalize_hypervolume(problem, hypervolume)
        num_nondominated = int(pareto_mask.sum().item())

        print(
            f"Iter {iteration + 1}/{config.n_iterations} | ND: {num_nondominated} | Hypervolume: {hypervolume:.4f}"
        )

        iteration_records.append(
            {
                "iteration": iteration + 1,
                "num_nondominated": num_nondominated,
                "hypervolume": float(hypervolume),
            }
        )

    end_time = time.time()
    print(f"\nBO loop finished in {end_time - start_time:.2f} seconds.")
    save_result(problem, config, iteration_records, train_obj)
