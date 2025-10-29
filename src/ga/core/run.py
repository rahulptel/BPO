import random
import time

import numpy as np
import torch
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from pymoo.operators.crossover.pntx import SinglePointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from .io import save_result


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def _normalize_hypervolume(hv, ideal_point):
    denom = np.abs(ideal_point).prod()
    return hv if denom == 0 else hv / denom


def get_algorithm(alg_cfg, n_objs):
    algorithm = None
    if alg_cfg.name == "nsga2":
        from pymoo.algorithms.moo.nsga2 import NSGA2

        algorithm = NSGA2(
            pop_size=alg_cfg.pop_size,
            sampling=BinaryRandomSampling(),
            crossover=SinglePointCrossover(),
            mutation=BitflipMutation(),
            eliminate_duplicates=alg_cfg.eliminate_duplicates,
        )
    elif alg_cfg.name == "nsga3":
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.util.ref_dirs import get_reference_directions

        ref_dirs = get_reference_directions(
            alg_cfg.ref_dir_method, int(n_objs), n_partitions=alg_cfg.n_partitions
        )

        algorithm = NSGA3(
            pop_size=alg_cfg.pop_size,
            ref_dirs=ref_dirs,
            sampling=BinaryRandomSampling(),
            crossover=SinglePointCrossover(),
            mutation=BitflipMutation(),
            eliminate_duplicates=alg_cfg.eliminate_duplicates,
        )

    elif alg_cfg.name == "smsemoa":
        from pymoo.algorithms.moo.sms import SMSEMOA

        algorithm = SMSEMOA(
            pop_size=alg_cfg.pop_size,
            sampling=BinaryRandomSampling(),
            crossover=SinglePointCrossover(),
            mutation=BitflipMutation(),
            eliminate_duplicates=alg_cfg.eliminate_duplicates,
        )

    elif alg_cfg.name == "ctaea":
        from pymoo.algorithms.moo.ctaea import CTAEA
        from pymoo.util.ref_dirs import get_reference_directions

        ref_dirs = get_reference_directions(
            alg_cfg.ref_dir_method, int(n_objs), n_partitions=alg_cfg.n_partitions
        )
        print(len(ref_dirs), "reference directions generated.")

        algorithm = CTAEA(
            ref_dirs=ref_dirs,
            sampling=BinaryRandomSampling(),
            crossover=SinglePointCrossover(),
            mutation=BitflipMutation(),
            eliminate_duplicates=alg_cfg.eliminate_duplicates,
        )

    return algorithm


def run_ga(problem, config):
    alg_cfg = config.algorithm
    prob_cfg = config.problem

    set_global_seed(alg_cfg.seed)

    if prob_cfg.ref_point is None:
        ref_point = problem.default_ref_point()
    else:
        ref_point = np.asarray(prob_cfg.ref_point, dtype=np.float64)
        if ref_point.size != problem.n_objs:
            raise ValueError(
                f"Ref point dimension {ref_point.size} does not match "
                f"n_objs={problem.n_objs}."
            )

    algorithm = get_algorithm(alg_cfg, problem.n_objs)
    termination = get_termination("time", alg_cfg.time)

    print(
        f"Running {alg_cfg.name} | pop_size: {alg_cfg.pop_size} | time: {alg_cfg.time} | "
        f"seed: {alg_cfg.seed}"
    )
    t0 = time.time()
    result = minimize(
        problem,
        algorithm,
        termination,
        seed=alg_cfg.seed,
        save_history=False,
        verbose=False,
    )
    total_time = time.time() - t0

    # Extract results (minimization space in pymoo)
    F = result.F if result.F is not None else np.empty((0, problem.n_objs))
    X = result.X if result.X is not None else np.empty((0, problem.n_items))

    # Convert to maximization space (profits) for BoTorch utilities
    Y_np = -F if F.size else F
    Y_t = torch.tensor(Y_np, dtype=torch.get_default_dtype())
    ref_point_t = torch.tensor(ref_point, dtype=torch.get_default_dtype())

    # Non-dominated mask using BoTorch (expects maximizing)
    if Y_t.numel() > 0:
        mask = is_non_dominated(Y_t)
        Y_t_nd = Y_t[mask]

        bd = FastNondominatedPartitioning(ref_point=ref_point_t, Y=Y_t_nd)
        hv = bd.compute_hypervolume().item()
        hv = _normalize_hypervolume(hv, problem.ideal_point())
        n_nd = mask.sum().item()

        mask = mask.cpu().numpy().astype(bool)
        Y_nd = Y_t_nd.cpu().numpy().tolist()
        X_nd = X[mask].tolist()
    else:
        X_nd, Y_nd = [], []
        hv, n_nd = 0.0, 0

    time_dict = {"optimization": total_time}

    print("Hypervolume (normalized): {:.6f}".format(hv))
    print("Number of non-dominated solutions: {}".format(n_nd))

    # Save ND sets (profits) and solutions
    save_result(problem, config, Y_nd, X_nd, hv, n_nd, ref_point, time_dict)
