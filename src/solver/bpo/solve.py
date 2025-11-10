import json
import time
from datetime import datetime

import torch
from omegaconf import OmegaConf

from utils import (
    OUTPUTS_DIR,
    compute_iteration_stats,
    get_dirichlet_distribution,
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
    def __init__(self, cfg, instance, scalarizer):
        self.cfg = cfg
        self.instance = instance
        self.scalarizer = scalarizer
        self.dirichlet = get_dirichlet_distribution(self.cfg.problem.n_objs)

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

    def save_result(self, iter_records, final_record, time_dict):
        cfg = self.cfg
        surrogate_dir_chain = self._surrogate_directory_chain(cfg.surrogate)
        acquisition_dir_chain = self._acquisition_directory_chain(self.cfg.acquisition)
        run_dir_chain = self._run_directory_chain(cfg)
        final_stats = None
        if final_record:
            final_stats = final_record[-1]
        elif iter_records:
            final_stats = iter_records[-1]

        results = {
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "iterations": iter_records,
            "final_record": final_record,
            "nondominated_solutions": final_stats["n_nd"] if final_stats else 0,
            "hv": final_stats["hv"] if final_stats else 0.0,
            "n_evaluations": self.scalarizer.n_evaluations,
            "time_dict": time_dict,
        }

        base_dir = OUTPUTS_DIR / "bpo"
        output_dir = base_dir / str(self.instance)
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
        time_dict["data_collection"] = 0
        # prefs = dirichlet_initial_design(
        #     self.cfg.n_initial_samples, self.instance.n_objs
        # )
        prefs = torch.empty((0, self.instance.n_objs), dtype=torch.get_default_dtype())
        objs = torch.empty((0, self.instance.n_objs), dtype=torch.get_default_dtype())

        data_collection_complete = True
        for _ in range(self.cfg.n_initial_samples):
            t0 = time.time()
            pref = self.dirichlet.sample((1,)).reshape(1, self.instance.n_objs)
            obj = self.scalarizer.evaluate(pref)
            time_dict["data_collection"] += time.time() - t0

            prefs = torch.cat([prefs, pref])
            objs = torch.cat([objs, obj])
            if time_dict["data_collection"] > self.cfg.time_limit:
                data_collection_complete = False
                break

        if data_collection_complete:
            print("Initial data generation complete.")

        return prefs, objs

    def run(self):
        set_global_seed(self.cfg.rseed)
        time_dict = {}

        print(f"Using reference point: {self.instance.reference_point.tolist()}")

        surrogate = build_surrogate(self.cfg.surrogate)
        print(f"Using surrogate: {self.cfg.surrogate.name}")

        if self.cfg.acquisition.name is None:
            self.cfg.acquisition.name = get_default_acquisition_name(
                self.cfg.surrogate.name
            )
        acquisition = build_acquisition(
            self.cfg.acquisition,
            n_objs=self.instance.n_objs,
            ref_point=self.instance.reference_point,
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
            # Check time limit (seconds) after updating iteration timing
            if time_dict["data_collection"] + time_dict["iterations"] >= float(
                self.cfg.time_limit
            ):
                print("Time limit reached. Stopping early.")
                break

            t0 = time.time()

            model = surrogate.fit(prefs, objs, time_dict)
            new_prefs = acquisition.generate_candidates(model, prefs, time_dict)
            new_objs = self.scalarizer.evaluate(new_prefs)

            time_dict["iterations"] += time.time() - t0

            prefs = torch.cat([prefs, new_prefs])
            objs = torch.cat([objs, new_objs])

        prefs_np = prefs.detach().cpu().numpy()
        objs_np = objs.detach().cpu().numpy()
        if self.scalarizer.maximize:
            objs_np = -objs_np

        t0 = time.time()
        iter_records = compute_iteration_stats(
            objs_np,
            self.instance.reference_point,
            ideal_point=self.instance.ideal_point,
            all_prefs=prefs_np,
            normalize_hypervolume=self.cfg.hypervolume.normalize,
            approx=self.cfg.hypervolume.approx,
            eps=self.cfg.hypervolume.eps,
            delta=self.cfg.hypervolume.delta,
            lib=self.cfg.hypervolume.lib,
            from_iteration=1,
            to_iteration=self.cfg.compute_stats_upto,
        )
        final_record = compute_iteration_stats(
            objs_np,
            self.instance.reference_point,
            ideal_point=self.instance.ideal_point,
            all_prefs=prefs_np,
            normalize_hypervolume=self.cfg.hypervolume.normalize,
            approx=self.cfg.hypervolume.approx,
            eps=self.cfg.hypervolume.eps,
            delta=self.cfg.hypervolume.delta,
            lib=self.cfg.hypervolume.lib,
            from_iteration=len(objs_np),
            to_iteration=len(objs_np),
        )
        time_dict["stats"] = time.time() - t0
        self.save_result(iter_records, final_record, time_dict)
        self.print_time_dict(time_dict)
        print("N evaluations:", self.scalarizer.n_evaluations)
