import random
import time

import numpy as np
import torch
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated

from scalarization.aug_cheby import AugChebyMOKPScalarizer

from .io import save_result


def _sample_dirichlet(n_points, dim):
    base = torch.ones(dim, dtype=torch.get_default_dtype())
    distribution = torch.distributions.dirichlet.Dirichlet(base)
    return distribution.sample((n_points,)).reshape(n_points, dim)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_hypervolume(unnorm_hv, ideal_point):
    if ideal_point is None:
        return unnorm_hv

    denom = torch.abs(ideal_point).prod().item()
    if denom == 0:
        return unnorm_hv

    return unnorm_hv / denom


def compute_iteration_stats(
    all_prefs,
    all_objs,
    ref_point,
    ideal_point,
    save_prefs=False,
    save_objs=False,
):
    records = []
    prev_n_nd = -1
    prev_hv = None

    for i, (prefs, objs) in enumerate(zip(all_prefs, all_objs)):
        unique_objs = torch.unique(objs, dim=0)
        pareto_mask = is_non_dominated(unique_objs)
        current_n_nd = int(pareto_mask.sum().item())

        if prev_n_nd == current_n_nd and prev_hv is not None:
            hv = prev_hv
        else:
            objs_nd = unique_objs[pareto_mask]
            bd = FastNondominatedPartitioning(ref_point=ref_point, Y=objs_nd)
            hv_val = bd.compute_hypervolume().item()
            hv = normalize_hypervolume(hv_val, ideal_point)
            prev_hv = hv
            prev_n_nd = current_n_nd

        print(
            f"Iter {i + 1}/{len(all_prefs)} | ND: {current_n_nd} | Hypervolume: {hv:.6f}"
        )

        record = {
            "iteration": i + 1,
            "n_nd": current_n_nd,
            "hv": float(hv),
        }

        if save_prefs or i == len(all_prefs) - 1:
            record["prefs"] = prefs.detach().cpu().tolist()
        if save_objs or i == len(all_objs) - 1:
            record["objs"] = objs.detach().cpu().tolist()

        records.append(record)

    return records


def solve(instance, cfg):
    set_global_seed(cfg.random.rseed)

    scalarizer = AugChebyMOKPScalarizer(
        instance,
        rho=cfg.problem.rho,
    )

    try:
        ref_point = torch.zeros(instance.n_objs, dtype=torch.get_default_dtype())
        ideal_point = torch.tensor(
            instance.ideal_point(),
            dtype=torch.get_default_dtype(),
        )

        print(f"Using reference point: {ref_point.tolist()}")

        print(f"Generating {cfg.random.n_initial_samples} initial data points...")
        time_dict = {"data_collection": 0.0, "iterations": 0.0}
        t0 = time.time()
        prefs = _sample_dirichlet(cfg.random.n_initial_samples, instance.n_objs)
        objs = scalarizer.evaluate(prefs)
        time_dict["data_collection"] = time.time() - t0
        print("Initial data generation complete.")

        all_prefs = [prefs]
        all_objs = [objs]

        max_iterations = int(cfg.random.n_iterations)
        batch_size = int(cfg.random.batch_size_q)
        time_limit = float(cfg.random.time_limit)

        print(f"Starting random loop for {max_iterations} iterations...")

        for iteration in range(max_iterations):
            if time_dict["data_collection"] + time_dict["iterations"] >= time_limit:
                print("Time limit reached; stopping early.")
                break

            t_iter_start = time.time()

            new_prefs = _sample_dirichlet(batch_size, instance.n_objs)
            new_objs = scalarizer.evaluate(new_prefs)

            prefs = torch.cat([prefs, new_prefs])
            objs = torch.cat([objs, new_objs])

            all_prefs.append(prefs)
            all_objs.append(objs)

            time_dict["iterations"] += time.time() - t_iter_start

        records = compute_iteration_stats(
            all_prefs,
            all_objs,
            ref_point,
            ideal_point,
            save_prefs=cfg.random.save_prefs,
            save_objs=cfg.random.save_objs,
        )

        print("N evaluations:", scalarizer.n_evaluations)

        save_result(
            instance,
            cfg,
            records,
            scalarizer.n_evaluations,
            ref_point,
            time_dict,
        )
    finally:
        scalarizer.close()
