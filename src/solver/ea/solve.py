import json
import random
import time
from datetime import datetime

import numpy as np
import torch
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from omegaconf import DictConfig, OmegaConf
from pymoo.core.callback import Callback
from pymoo.operators.crossover.pntx import SinglePointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from utils import OUTPUTS_DIR, compute_hypervolume


class GenerationTrackingCallback(Callback):
    def __init__(self, start_time):
        super().__init__()
        self.n_generation = 0
        self.start_time = start_time
        self.data["records"] = []

    def notify(self, algorithm):
        self.n_generation += 1


class EASolver:
    def __init__(self, cfg, base_instance, pymoo_instance):
        self.cfg = cfg
        self.base_instance = base_instance
        self.pymoo_instance = pymoo_instance

    @staticmethod
    def _set_global_seed(seed):
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def _normalize_hypervolume(hv, ideal_point):
        denom = np.abs(ideal_point).prod()
        return hv if denom == 0 else hv / denom

    @staticmethod
    def _get_nondominated(Y):
        Y_t = torch.tensor(Y, dtype=torch.get_default_dtype())
        mask = is_non_dominated(Y_t)
        return mask, Y_t[mask]

    def _build_algorithm(self):
        alg_cfg = self.cfg.algorithm
        n_objs = self.cfg.problem.n_objs

        if alg_cfg.name == "nsga2":
            from pymoo.algorithms.moo.nsga2 import NSGA2

            return NSGA2(
                pop_size=alg_cfg.pop_size,
                sampling=BinaryRandomSampling(),
                crossover=SinglePointCrossover(),
                mutation=BitflipMutation(),
                eliminate_duplicates=alg_cfg.eliminate_duplicates,
            )
        if alg_cfg.name == "nsga3":
            from pymoo.algorithms.moo.nsga3 import NSGA3
            from pymoo.util.ref_dirs import get_reference_directions

            ref_dirs = get_reference_directions(
                alg_cfg.ref_dir_method, int(n_objs), n_partitions=alg_cfg.n_partitions
            )
            print(len(ref_dirs), "reference directions generated.")

            return NSGA3(
                pop_size=alg_cfg.pop_size,
                ref_dirs=ref_dirs,
                sampling=BinaryRandomSampling(),
                crossover=SinglePointCrossover(),
                mutation=BitflipMutation(),
                eliminate_duplicates=alg_cfg.eliminate_duplicates,
            )
        if alg_cfg.name == "smsemoa":
            from pymoo.algorithms.moo.sms import SMSEMOA

            return SMSEMOA(
                pop_size=alg_cfg.pop_size,
                sampling=BinaryRandomSampling(),
                crossover=SinglePointCrossover(),
                mutation=BitflipMutation(),
                eliminate_duplicates=alg_cfg.eliminate_duplicates,
            )
        if alg_cfg.name == "ctaea":
            from pymoo.algorithms.moo.ctaea import CTAEA
            from pymoo.util.ref_dirs import get_reference_directions

            ref_dirs = get_reference_directions(
                alg_cfg.ref_dir_method, int(n_objs), n_partitions=alg_cfg.n_partitions
            )
            print(len(ref_dirs), "reference directions generated.")

            return CTAEA(
                ref_dirs=ref_dirs,
                sampling=BinaryRandomSampling(),
                crossover=SinglePointCrossover(),
                mutation=BitflipMutation(),
                eliminate_duplicates=alg_cfg.eliminate_duplicates,
            )

        raise ValueError(f"Unknown EA algorithm '{alg_cfg.name}'")

    @staticmethod
    def _algorithm_directory_chain(cfg):
        time_str = str(cfg.algorithm.time)
        time_str = time_str.replace(":", "-").replace("/", "-").replace(" ", "_")
        return [
            ("algorithm", str(cfg.algorithm.name).lower()),
            ("pop_size", cfg.algorithm.pop_size),
            ("time", time_str),
        ]

    def save_result(self, y_sol_nd, x_sol_nd, hv, n_nd, n_generations, time_dict):
        results = {
            "cfg": OmegaConf.to_container(self.cfg, resolve=True),
            "x_sol_nd": x_sol_nd,
            "y_sol_nd": y_sol_nd,
            "hypervolume": hv,
            "n_nd": n_nd,
            "n_generations": n_generations,
            "time_dict": time_dict,
        }

        output_dir = (
            OUTPUTS_DIR
            / "ea"
            / f"{str(self.base_instance)}_seed-{self.cfg.algorithm.seed}"
        )
        for key, value in self._algorithm_directory_chain(self.cfg):
            output_dir /= f"{key}-{value}"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = output_dir / f"run_ea_{timestamp}.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"Saved EA results to {output_path}")
        return output_path

    def run(self):
        self._set_global_seed(self.cfg.algorithm.seed)

        ref_point = self.base_instance.reference_point
        ref_point_t = torch.tensor(ref_point, dtype=torch.get_default_dtype())

        ideal_point = self.base_instance.ideal_point
        algorithm = self._build_algorithm()
        termination = get_termination("time", self.cfg.algorithm.time)

        print(
            f"Running {self.cfg.algorithm.name} |"
            f"pop_size: {self.cfg.algorithm.pop_size} |"
            f"time: {self.cfg.algorithm.time} |"
            f"seed: {self.cfg.algorithm.seed} |"
            f"Track generations: {self.cfg.track_generations}"
        )

        t0 = time.time()
        callback = (
            GenerationTrackingCallback(t0) if self.cfg.track_generations else None
        )
        minimize_kwargs = {
            "save_history": False,
            "verbose": False,
            "seed": self.cfg.algorithm.seed,
        }
        if callback is not None:
            minimize_kwargs["callback"] = callback

        result = minimize(
            self.pymoo_instance,
            algorithm,
            termination,
            **minimize_kwargs,
        )
        total_time = time.time() - t0
        time_dict = {"optimization": total_time}
        print("Optimization finished in {:.2f} seconds.".format(total_time))

        # Get nondominated expects objective vector in the maximization form
        mask_t, Y_nd_t = self._get_nondominated(-result.F)
        mask = mask_t.cpu().numpy().astype(bool)
        n_nd = int(mask.sum())

        hv = 0.0
        Y_nd = []
        X_nd = []
        if n_nd > 0:
            hv = compute_hypervolume(Y_nd_t, ref_point_t, ideal_point=ideal_point)
            Y_nd = Y_nd_t.cpu().numpy().tolist()
            X_nd = result.X[mask].tolist() if result.X is not None else None
        n_generations = -1 if callback is None else callback.n_generation

        print("Hypervolume (normalized): {:.6f}".format(hv))
        print("Number of non-dominated solutions: {}".format(n_nd))
        self.save_result(
            Y_nd,
            X_nd,
            hv,
            n_nd,
            n_generations,
            time_dict,
        )
