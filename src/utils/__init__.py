import random
from pathlib import Path

import gurobipy as gp
import numpy as np
import pygmo as pg
import torch
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

SRC_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = SRC_DIR.parent / "outputs"


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_gurobi_env(output_flag=0, threads=1, time_limit=100):
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", output_flag)
    env.setParam("Threads", threads)
    env.setParam("TimeLimit", time_limit)

    return env


def get_dirichlet_distribution(dim):
    base = torch.ones(dim, dtype=torch.get_default_dtype())
    distribution = torch.distributions.dirichlet.Dirichlet(base)
    return distribution


# def compute_hypervolume(Y_nd, ref_point, ideal_point=None, normalize=True):
#     bd = FastNondominatedPartitioning(ref_point=ref_point, Y=Y_nd)
#     hv_val = bd.compute_hypervolume().item()
#     if normalize and ideal_point is not None:
#         return normalize_hypervolume(hv_val, ideal_point)
#     return float(hv_val)


def compute_hypervolume(
    points,
    ref_point,
    ideal_point=None,
    normalize=True,
    approx=False,
    eps=0.1,
    delta=0.1,
):
    # pygmo expects points in minimization form
    hv = pg.hypervolume(points)
    hv_val = (
        hv.compute(ref_point, hv_algo=pg.bf_fpras(eps=eps, delta=delta))
        if approx
        else hv.compute(ref_point)
    )
    hv_val = normalize_hypervolume(hv_val, ideal_point) if normalize else hv_val

    return hv_val


def normalize_hypervolume(unnorm_hv, ideal_point):
    if ideal_point is None:
        return unnorm_hv

    denom = np.abs(ideal_point).prod()
    if denom == 0:
        return unnorm_hv

    return float(unnorm_hv / denom)


def compute_iteration_stats(
    all_objs,
    ref_point,
    ideal_point=None,
    all_prefs=None,
    normalize_hypervolume=True,
    approx=False,
    eps=0.1,
    delta=0.1,
    from_iteration=1,
    to_iteration=100,
):
    """
    Expect all_objs to be in minimization form
    """
    records, prev_n_nd = [], -1

    if all_prefs is not None:
        if isinstance(all_prefs, torch.Tensor):
            all_prefs = all_prefs.detach().cpu().numpy()
        if isinstance(all_prefs, np.ndarray):
            all_prefs = all_prefs.tolist()

    to_iteration = min(len(all_objs), to_iteration)
    for i in range(from_iteration, to_iteration + 1):
        objs = all_objs[:i]
        objs = np.unique(objs, axis=0)
        nd_idx = NonDominatedSorting().do(objs, only_non_dominated_front=True)

        n_nd = len(nd_idx)
        if n_nd > 0:
            objs = objs[nd_idx]

        if prev_n_nd > 0 and n_nd == prev_n_nd:
            # No change in ND front, skip HV computation
            hv = records[-1]["hv"]
        else:
            hv = compute_hypervolume(
                objs,
                ref_point,
                ideal_point=ideal_point,
                normalize=normalize_hypervolume,
                approx=approx,
                eps=eps,
                delta=delta,
            )

        prev_n_nd = n_nd
        iteration_record = {
            "n_evaluation": i,
            "n_nd": n_nd,
            "hv": hv,
        }

        records.append(iteration_record)
        print(f"Iter {i}/{len(all_objs)} | ND: {n_nd} | " f"Hypervolume: {hv}")

    return records
