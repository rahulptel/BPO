import random
import time

import numpy as np
import torch
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated

from bpo.acquisition import ACQUISITION_REGISTRY, AcquisitionConfig
from bpo.core.model import available_surrogates

from .io import save_result
from .model import build_surrogate


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_reference_point(problem, cfg):
    if cfg.problem.ref_point is None:
        ref_point = problem.default_ref_point()
    else:
        ref_point = torch.tensor(cfg.problem.ref_point, dtype=torch.get_default_dtype())
        if ref_point.numel() != problem.n_objectives():
            raise ValueError(
                f"Ref point dimension {ref_point.numel()} does not match "
                f"n_objs={problem.n_objectives()}."
            )

    return ref_point


def get_default_acquisition_name(surrogate_name):
    if surrogate_name in ("gp", "ibnn"):
        return "qlogehvi"
    elif surrogate_name == "none":
        return "random"
    else:
        raise ValueError(f"Cannot infer acquisition for surrogate '{surrogate_name}'.")


def build_acquisition(problem, cfg, ref_point):
    bounds = problem.lambda_bounds()
    equality_constraints = problem.lambda_equality_constraints()

    if cfg.acquisition.name is None:
        acquisition_name = get_default_acquisition_name(cfg.surrogate.name)
        cfg.acquisition.name = acquisition_name

    acq_cfg = AcquisitionConfig(
        ref_point=ref_point,
        bounds=bounds,
        batch_size=cfg.acquisition.batch_size_q,
        num_restarts=cfg.bo.num_restarts,
        raw_samples=cfg.bo.raw_samples,
        sequential=cfg.acquisition.sequential,
        equality_constraints=equality_constraints,
        mc_samples=cfg.acquisition.mc_samples,
        rseed=cfg.bo.rseed,
    )

    if cfg.acquisition.name not in ACQUISITION_REGISTRY:
        raise ValueError(f"Unknown acquisition function '{cfg.acquisition.name}'")

    return ACQUISITION_REGISTRY[cfg.acquisition.name](acq_cfg)


def normalize_hypervolume(unnorm_hv, ideal_point):
    if ideal_point is None:
        return unnorm_hv

    denom = torch.abs(ideal_point).prod().item()
    if denom == 0:
        return unnorm_hv

    return unnorm_hv / denom


def compute_iteration_stats(all_prefs, all_objs, ref_point, ideal_point, n_iterations):
    records = []
    for i, (prefs, objs) in enumerate(zip(all_prefs, all_objs)):
        pareto_mask = is_non_dominated(objs)
        objs_nd = objs[pareto_mask]

        bd = FastNondominatedPartitioning(ref_point=ref_point, Y=objs_nd)
        hv = bd.compute_hypervolume().item()
        hv = normalize_hypervolume(hv, ideal_point)
        n_nd = int(pareto_mask.sum().item())

        print(f"Iter {i + 1}/{n_iterations} | ND: {n_nd} | " f"Hypervolume: {hv:.4f}")
        records.append(
            {
                "iteration": i + 1,
                "n_nd": n_nd,
                "hv": float(hv),
                "prefs": prefs.cpu().tolist(),
                "objs": objs.cpu().tolist(),
            }
        )

    return records


def print_time_dict(time_dict):
    print(f"\nBO loop timing: ")
    print(f"\tData collection: {time_dict['data_collection']:.2f} seconds.")
    print(f"\tIterations: {time_dict['iterations']:.2f} seconds.")
    if "surrogate_training" in time_dict:
        print(f"\t\tSurrogate training: {time_dict['surrogate_training']:.2f} seconds.")
    if "acquisition_optimization" in time_dict:
        print(
            f"\t\tAcquisition optimization: {time_dict['acquisition_optimization']:.2f} seconds."
        )
    if "partitioning" in time_dict:
        print(f"\t\t\tPartitioning: {time_dict['partitioning']:.2f} seconds.")


def run_bo(problem, cfg):
    set_global_seed(cfg.bo.rseed)
    time_dict = {}

    ref_point = get_reference_point(problem, cfg)
    print(f"Using reference point: {ref_point.tolist()}")

    assert cfg.surrogate.name.lower() in available_surrogates()
    surrogate = build_surrogate(cfg.surrogate.name, cfg.surrogate)
    print(f"Using surrogate: {cfg.surrogate.name}")

    acquisition = build_acquisition(problem, cfg, ref_point)
    print(
        f"Using acquisition: {acquisition.__class__.__name__} "
        f"| rseed: {cfg.bo.rseed}"
    )

    print(f"Generating {cfg.bo.n_initial_samples} initial data points...")
    t0 = time.time()
    prefs = problem.initial_design(cfg.bo.n_initial_samples)
    objs = problem.evaluate(prefs)
    time_dict["data_collection"] = time.time() - t0
    print("Initial data generation complete.")

    print(f"Starting BO loop for {cfg.bo.n_iterations} iterations...")
    time_dict["iterations"] = 0.0
    all_prefs, all_objs = [], []
    for _ in range(cfg.bo.n_iterations):
        t0 = time.time()

        model = surrogate.fit(prefs, objs, time_dict)
        new_prefs = acquisition.generate_candidates(model, prefs, objs, time_dict)
        new_objs = problem.evaluate(new_prefs)

        time_dict["iterations"] += time.time() - t0

        prefs = torch.cat([prefs, new_prefs])
        objs = torch.cat([objs, new_objs])

        all_prefs.append(prefs)
        all_objs.append(objs)

    records = compute_iteration_stats(
        all_prefs, all_objs, ref_point, problem.ideal_point(), cfg.bo.n_iterations
    )
    print_time_dict(time_dict)

    save_result(
        problem,
        cfg,
        records,
        ref_point,
        time_dict,
    )
