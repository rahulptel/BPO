import random
from pathlib import Path

import gurobipy as gp
import numpy as np
import torch
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated

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


def compute_hypervolume(Y_nd, ref_point, ideal_point=None, normalize=True):
    bd = FastNondominatedPartitioning(ref_point=ref_point, Y=Y_nd)
    hv_val = bd.compute_hypervolume().item()
    if normalize and ideal_point is not None:
        return normalize_hypervolume(hv_val, ideal_point)
    return float(hv_val)


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
    n_evaluations=None,
    all_prefs=None,
    save_prefs=False,
    save_objs=False,
):
    """
    Expect all_objs to be tensor
    """
    records, prev_n_nd = [], -1

    # Save reference point to tensor
    if type(ref_point) != torch.Tensor:
        ref_point = torch.tensor(ref_point, dtype=torch.get_default_dtype())

    # Save preference to list
    if all_prefs is not None:
        if type(all_prefs) == torch.Tensor:
            all_prefs = all_prefs.detach().cpu().numpy()
        if type(all_prefs) == np.ndarray:
            all_prefs = all_prefs.tolist()

    # Save objs to list for saving

    n_evaluations = len(all_objs) if n_evaluations is None else n_evaluations
    for i in range(1, n_evaluations + 1):
        if i <= 100 or i == n_evaluations:
            objs = all_objs[:i]
            unique_objs = torch.unique(objs, dim=0)
            pareto_mask = is_non_dominated(unique_objs)
            n_nd = int(pareto_mask.sum().item())
            if prev_n_nd > 0 and pareto_mask.sum().item() == prev_n_nd:
                # No change in ND front, skip HV computation
                hv = records[-1]["hv"]
            else:
                objs_nd = unique_objs[pareto_mask]
                hv = compute_hypervolume(objs_nd, ref_point, ideal_point=ideal_point)
                prev_n_nd = n_nd

            iteration_record = {
                "n_evaluation": i,
                "n_nd": n_nd,
                "hv": hv,
            }

            if save_prefs or i == len(all_objs) - 1:
                iteration_record["prefs"] = all_prefs[:i]
            if save_objs or i == len(all_objs) - 1:
                iteration_record["objs"] = objs.detach().cpu().tolist()
            records.append(iteration_record)
            print(f"Iter {i}/{n_evaluations} | ND: {n_nd} | " f"Hypervolume: {hv:.6f}")

    return records
