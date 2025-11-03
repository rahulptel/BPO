import json
import time
from datetime import datetime

import torch
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from omegaconf import OmegaConf

from scalarization.aug_cheby import AugChebyMOKPScalarizer
from utils import (
    OUTPUTS_DIR,
    compute_hypervolume,
    compute_iteration_stats,
    dirichlet_initial_design,
    normalize_hypervolume,
    set_global_seed,
)

from .acquisition import build_acquisition
from .surrogate import build_surrogate


def get_default_acquisition_name(surrogate_name):
    if surrogate_name in ("gp", "ibnn"):
        return "qlogehvi"
    elif surrogate_name == "none":
        return "random"
    else:
        raise ValueError(f"Cannot infer acquisition for surrogate '{surrogate_name}'.")


class BPOSolver:
    def __init__(self, cfg, problem):
        self.cfg = cfg
        self.problem = problem
        self.scalarizer = None

        self._init_scalarizer()

    def _init_scalarizer(self):
        if self.cfg.scalarization.name == "aug_cheby":
            self.scalarizer = AugChebyMOKPScalarizer(
                self.problem,
                rho=self.cfg.scalarization.rho,
            )
        else:
            raise ValueError(
                f"Unknown scalarization name: {self.cfg.scalarization.name}"
            )

    @staticmethod
    def _surrogate_directory_chain(config):
        dir_chain = [("surr", config.name)]
        return dir_chain

    @staticmethod
    def _acquisition_directory_chain(config):
        dir_chain = [("acq", config.name)]
        if config.batch_size_q is not None:
            dir_chain.append(("batch_size_q", config.batch_size_q))
        if config.num_restarts is not None:
            dir_chain.append(("num_restarts", config.num_restarts))
        if config.raw_samples is not None:
            dir_chain.append(("raw_samples", config.raw_samples))
        if config.sequential is not None:
            dir_chain.append(("sequential", config.sequential))
        if config.mc_samples is not None:
            dir_chain.append(("mc_samples", config.mc_samples))
        return dir_chain

    @staticmethod
    def _run_directory_chain(config):
        dir_chain = []
        if config.n_initial_samples is not None:
            dir_chain.append(("n_init", config.n_initial_samples))
        if config.n_iterations is not None:
            dir_chain.append(("n_iter", config.n_iterations))
        if config.time_limit is not None:
            dir_chain.append(("time", config.time_limit))
        if config.n_iterations is not None:
            dir_chain.append(("rseed", config.rseed))

        return dir_chain

    def _get_default_acquisition_name(self, surrogate_name):
        if surrogate_name in ("gp", "ibnn"):
            return "qlogehvi"
        else:
            raise ValueError(
                f"Cannot infer acquisition for surrogate '{surrogate_name}'."
            )

    @staticmethod
    def print_time_dict(time_dict):
        print(f"\nBO loop timing: ")
        print(f"\tData collection: {time_dict['data_collection']:.2f} seconds.")
        print(f"\tIterations: {time_dict['iterations']:.2f} seconds.")
        if "surrogate_training" in time_dict:
            print(
                f"\t\tSurrogate training: {time_dict['surrogate_training']:.2f} seconds."
            )
        if "acquisition_optimization" in time_dict:
            print(
                f"\t\tAcquisition optimization: {time_dict['acquisition_optimization']:.2f} seconds."
            )
        if "partitioning" in time_dict:
            print(f"\t\t\tPartitioning: {time_dict['partitioning']:.2f} seconds.")

    def save_result(
        self,
        records,
        time_dict,
    ):
        cfg = self.cfg
        surrogate_dir_chain = self._surrogate_directory_chain(cfg.surrogate)
        acquisition_dir_chain = self._acquisition_directory_chain(self.cfg.acquisition)
        run_dir_chain = self._run_directory_chain(cfg)

        results = {
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "iterations": records,
            "nondominated_solutions": records[-1]["n_nd"],
            "hv": records[-1]["hv"],
            "n_evaluations": self.scalarizer.n_evaluations,
            "time_dict": time_dict,
        }

        base_dir = OUTPUTS_DIR / "bpo"
        output_dir = base_dir / str(self.problem)
        for name, value in surrogate_dir_chain:
            output_dir /= f"{name}-{value}"
        for name, value in acquisition_dir_chain:
            output_dir /= f"{name}-{value}"
        for name, value in run_dir_chain:
            output_dir /= f"{name}-{value}"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = output_dir / f"run_bo_{timestamp}.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"Saved BO results to {output_path}")
        return output_path

    def prepare_initial_training_data(self, time_dict):
        print(f"Generating {self.cfg.n_initial_samples} initial data points...")
        t0 = time.time()
        prefs = dirichlet_initial_design(
            self.cfg.n_initial_samples, self.problem.n_objs
        )
        objs = self.scalarizer.evaluate(prefs)
        time_dict["data_collection"] = time.time() - t0
        print("Initial data generation complete.")
        return prefs, objs

    def run(self):
        set_global_seed(self.cfg.rseed)
        time_dict, records = {}, []

        print(f"Using reference point: {self.problem.reference_point.tolist()}")

        surrogate = build_surrogate(self.cfg.surrogate)
        print(f"Using surrogate: {self.cfg.surrogate.name}")

        if self.cfg.acquisition.name is None:
            self.cfg.acquisition.name = get_default_acquisition_name(
                self.cfg.surrogate.name
            )
        acquisition = build_acquisition(
            self.cfg.acquisition,
            n_objs=self.problem.n_objs,
            ref_point=self.problem.reference_point,
            rseed=self.cfg.rseed,
        )
        print(
            f"Using acquisition: {self.cfg.acquisition.name} "
            f"| rseed: {self.cfg.rseed}"
        )

        prefs, objs = self.prepare_initial_training_data(time_dict)
        print(f"Starting BO loop for {self.cfg.n_iterations} iterations...")
        time_dict["iterations"] = 0.0
        for _ in range(self.cfg.n_iterations):
            t0 = time.time()

            model = surrogate.fit(prefs, objs, time_dict)
            new_prefs = acquisition.generate_candidates(model, prefs, time_dict)
            new_objs = self.scalarizer.evaluate(new_prefs)

            time_dict["iterations"] += time.time() - t0

            prefs = torch.cat([prefs, new_prefs])
            objs = torch.cat([objs, new_objs])

            # Check time limit (seconds) after updating iteration timing
            if time_dict["data_collection"] + time_dict["iterations"] >= float(
                self.cfg.time_limit
            ):
                print("Time limit reached; stopping early.")
                break

        records = compute_iteration_stats(
            objs,
            self.problem.reference_point,
            self.problem.ideal_point,
            len(objs),
            all_prefs=prefs,
            save_objs=self.cfg.save_objs,
            save_prefs=self.cfg.save_prefs,
        )
        self.save_result(records, time_dict)
        self.print_time_dict(time_dict)
        print("N evaluations:", self.scalarizer.n_evaluations)
        self.scalarizer.n_evaluations = 0
