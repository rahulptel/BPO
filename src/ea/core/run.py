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


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def _normalize_hypervolume(hv, ideal_point):
    denom = np.abs(ideal_point).prod()
    return hv if denom == 0 else hv / denom


def _get_hypervolume(F, ref_point, ideal_point):
    Y_np = -np.asarray(F)
    Y_t = torch.tensor(Y_np, dtype=torch.get_default_dtype())
    mask = is_non_dominated(Y_t)
    n_nd = int(mask.sum().item())

    hv = 0.0
    if n_nd > 0:
        bd = FastNondominatedPartitioning(
            ref_point=ref_point,
            Y=Y_t[mask],
        )
        hv_val = bd.compute_hypervolume().item()
        hv = _normalize_hypervolume(hv_val, ideal_point)

    return (
        mask.cpu().numpy().astype(bool),
        n_nd,
        hv,
    )


def _extract_nd_sets(F, X, ref_point, ideal_point):
    if F is None:
        return [], [], 0.0, 0
    F_arr = np.asarray(F)
    if F_arr.size == 0:
        return [], [], 0.0, 0
    mask, n_nd, hv = _get_hypervolume(F_arr, ref_point, ideal_point)
    if n_nd == 0:
        return [], [], 0.0, 0

    Y_nd = (-F_arr[mask]).tolist()
    X_nd = []
    if X is not None:
        X_arr = np.asarray(X)
        if X_arr.shape[0] == mask.shape[0]:
            X_nd = X_arr[mask].tolist()
    return Y_nd, X_nd, float(hv), int(n_nd)


class GenerationTrackingCallback(Callback):
    def __init__(self, start_time):
        super().__init__()
        self.generation = 0
        self.start_time = start_time
        self.data["records"] = []

    def notify(self, algorithm):
        self.generation += 1

        pop = getattr(algorithm, "pop", None)
        if pop is not None:
            F_val = pop.get("F")
            X_val = pop.get("X")
        elapsed = time.time() - self.start_time

        self.data["records"].append(
            {
                "generation": int(self.generation),
                "time": float(elapsed),
                "F": F_val,
                "X": X_val,
            }
        )


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


def run_ea(problem, config):
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
    ref_point_t = torch.tensor(ref_point, dtype=torch.get_default_dtype())
    ideal_point = problem.ideal_point()

    algorithm = get_algorithm(alg_cfg, problem.n_objs)
    termination = get_termination("time", alg_cfg.time)

    print(
        f"Running {alg_cfg.name} | pop_size: {alg_cfg.pop_size} | time: {alg_cfg.time} | \n"
        f"seed: {alg_cfg.seed} | Track generations: {config.track_generations}"
    )
    t0 = time.time()
    callback = GenerationTrackingCallback(t0) if config.track_generations else None
    result = minimize(
        problem,
        algorithm,
        termination,
        seed=alg_cfg.seed,
        save_history=False,
        verbose=False,
        callback=callback,
    )
    total_time = time.time() - t0
    time_dict = {"optimization": total_time}
    print("Optimization finished in {:.2f} seconds.".format(total_time))

    generation_records, F, X = [], None, None
    if callback is not None:
        generation_records = callback.data.get("records", [])
        for gen, record in enumerate(generation_records):
            F, X = record.get("F"), record.get("X")
            n_nd = 0
            hv = 0.0
            if F is not None:
                mask, n_nd, hv = _get_hypervolume(F, ref_point_t, ideal_point)

            record["n_nd"] = int(n_nd)
            record["hypervolume"] = float(hv)

            print(
                f"Generation {gen + 1:3d} | "
                f"Time Elapsed: {record['time']:.2f}s | "
                f"Non-dominated Solutions: {n_nd:4d} | "
                f"Hypervolume (normalized): {hv:.6f}"
            )

            record.pop("F", None)
            record.pop("X", None)
    else:
        F, X = result.F, result.X
        mask, n_nd, hv = _get_hypervolume(F, ref_point_t, ideal_point)

    Y_nd, X_nd = F[mask], X[mask] if F is not None else None, None
    print("Hypervolume (normalized): {:.6f}".format(hv))
    print("Number of non-dominated solutions: {}".format(n_nd))

    # Save ND sets (profits) and solutions
    save_result(
        problem,
        config,
        Y_nd,
        X_nd,
        hv,
        n_nd,
        ref_point,
        time_dict,
        generation_records,
    )
