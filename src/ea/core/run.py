import random
import time

import numpy as np
import torch
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from pymoo.core.callback import Callback
from pymoo.operators.crossover.pntx import SinglePointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from .io import save_result


class GenerationTrackingCallback(Callback):
    def __init__(self, start_time):
        super().__init__()
        self.n_generation = 0
        self.start_time = start_time
        self.data["records"] = []

    def notify(self, algorithm):
        self.n_generation += 1


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_reference_point(cfg, problem):
    if cfg.problem.ref_point is None:
        ref_point = problem.default_ref_point()
    else:
        ref_point = np.asarray(cfg.problem.ref_point, dtype=np.float64)
        if ref_point.size != problem.n_objs:
            raise ValueError(
                f"Ref point dimension {ref_point.size} does not match "
                f"n_objs={problem.n_objs}."
            )
    ref_point_t = torch.tensor(ref_point, dtype=torch.get_default_dtype())

    return ref_point, ref_point_t


def normalize_hypervolume(hv, ideal_point):
    denom = np.abs(ideal_point).prod()
    return hv if denom == 0 else hv / denom


def get_nondominated(Y):
    Y_t = torch.tensor(Y, dtype=torch.get_default_dtype())
    mask = is_non_dominated(Y_t)
    return mask, Y_t[mask]


def get_hypervolume(Y_nd, ref_point, ideal_point):
    hv = 0.0
    bd = FastNondominatedPartitioning(
        ref_point=ref_point,
        Y=Y_nd,
    )
    hv_val = bd.compute_hypervolume().item()
    hv = normalize_hypervolume(hv_val, ideal_point)

    return hv


def _extract_nd_sets(F, X, ref_point, ideal_point):
    if F is None:
        return [], [], 0.0, 0

    mask, Y_nd = get_nondominated(-F)
    n_nd = mask.sum().item()
    if n_nd == 0:
        return [], [], 0.0, 0

    hv = get_hypervolume(F, ref_point, ideal_point)
    Y_nd = (-F[mask]).tolist()
    X_nd = X[mask].tolist() if X is not None else None

    return Y_nd, X_nd, float(hv), int(n_nd)


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


def run_ea(problem, cfg):
    set_global_seed(cfg.algorithm.seed)

    ref_point, ref_point_t = get_reference_point(cfg, problem)
    ideal_point = problem.ideal_point()
    algorithm = get_algorithm(cfg.algorithm, problem.n_objs)
    termination = get_termination("time", cfg.algorithm.time)

    print(
        f"Running {cfg.algorithm.name} |"
        f"pop_size: {cfg.algorithm.pop_size} |"
        f"time: {cfg.algorithm.time} |"
        f"seed: {cfg.algorithm.seed} |"
        f"Track generations: {cfg.track_generations}"
    )

    t0 = time.time()

    callback = GenerationTrackingCallback(t0) if cfg.track_generations else None
    minimize_kwargs = {
        "save_history": False,
        "verbose": False,
        "seed": cfg.algorithm.seed,
    }
    if callback is not None:
        minimize_kwargs["callback"] = callback

    result = minimize(
        problem,
        algorithm,
        termination,
        **minimize_kwargs,
    )
    total_time = time.time() - t0
    time_dict = {"optimization": total_time}
    print("Optimization finished in {:.2f} seconds.".format(total_time))

    n_generations = -1 if callback is None else callback.n_generation
    mask, Y_nd = get_nondominated(-result.F)
    mask = mask.cpu().numpy().astype(bool)
    X_nd = result.X[mask].tolist() if result.X is not None else None
    n_nd = mask.sum().item()
    hv = 0 if n_nd == 0 else get_hypervolume(Y_nd, ref_point_t, ideal_point)
    Y_nd = Y_nd.cpu().numpy().tolist()

    print("Hypervolume (normalized): {:.6f}".format(hv))
    print("Number of non-dominated solutions: {}".format(n_nd))

    # Save ND sets (profits) and solutions
    save_result(
        problem,
        cfg,
        Y_nd,
        X_nd,
        hv,
        n_nd,
        n_generations,
        ref_point.tolist(),
        time_dict,
    )
