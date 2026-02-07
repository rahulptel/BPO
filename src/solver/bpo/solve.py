import json
import time
from datetime import datetime

import torch
from omegaconf import OmegaConf

from utils import (
    OUTPUTS_DIR,
    compute_iteration_stats,
    set_global_seed,
)

from .acquisition import build_acquisition
from .surrogate import build_surrogate


def get_default_acquisition_name(surrogate_name):
    if surrogate_name == "gp":
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
        self.dtype = self._resolve_runtime_dtype()
        self.device = self._resolve_runtime_device()
        self.dirichlet = torch.distributions.dirichlet.Dirichlet(
            torch.ones(
                self.cfg.problem.n_objs,
                dtype=self.dtype,
                device=self.device,
            )
        )

    def _resolve_runtime_dtype(self):
        runtime_cfg = getattr(self.cfg, "runtime", None)
        dtype_name = (
            str(getattr(runtime_cfg, "bo_dtype", "float64")).lower()
            if runtime_cfg is not None
            else "float64"
        )
        if dtype_name == "float32":
            return torch.float32
        if dtype_name == "float64":
            return torch.float64
        raise ValueError(
            f"Unsupported runtime.bo_dtype='{dtype_name}'. Use float32 or float64."
        )

    def _resolve_runtime_device(self):
        runtime_cfg = getattr(self.cfg, "runtime", None)
        if runtime_cfg is None:
            return torch.device("cpu")

        requested = str(getattr(runtime_cfg, "bo_device", "cpu")).lower()
        fallback = bool(getattr(runtime_cfg, "cuda_fallback_to_cpu", True))

        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if requested == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if fallback:
                print("CUDA requested but unavailable; falling back to CPU.")
                return torch.device("cpu")
            raise RuntimeError("CUDA requested but not available.")

        if requested != "cpu":
            raise ValueError(
                f"Unsupported runtime.bo_device='{requested}'. Use cpu, cuda, or auto."
            )
        return torch.device("cpu")

    @staticmethod
    def _surrogate_directory_chain(config):
        dir_chain = [("surr", config.name)]
        kernel = getattr(config, "kernel", None)
        if kernel is not None:
            dir_chain.append(("kernel", kernel))
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
        if getattr(config, "maxiter", None) is not None:
            dir_chain.append(("maxiter", config.maxiter))
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
        if surrogate_name == "gp":
            return "qlogehvi"
        else:
            raise ValueError(
                f"Cannot infer acquisition for surrogate '{surrogate_name}'."
            )

    @staticmethod
    def print_time_dict(time_dict):
        print(f"\nBO loop timing: ")
        if len(time_dict["data_collection"]) > 0:
            print(f"\tData collection: {time_dict['data_collection'][-1]:.2f} seconds.")
            if len(time_dict["iterations"]) > 0:
                print(f"\tIterations: {time_dict['iterations'][-1]:.2f} seconds.")
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

    def save_result(self, prefs, objs, iter_records, final_record, time_dict):
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
            "prefs": prefs.tolist(),
            "objs": objs.tolist(),
            "n_evaluations": self.scalarizer.n_evaluations,
            "n_evaluations_saved": len(objs),
            "n_evaluations_solver": self.scalarizer.n_evaluations,
            "iter_records": iter_records,
            "final_record": final_record,
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
        time_dict["data_collection"] = []
        data_collection = 0
        prefs = torch.empty(
            (0, self.instance.n_objs),
            dtype=self.dtype,
            device=self.device,
        )
        objs = torch.empty(
            (0, self.instance.n_objs),
            dtype=self.dtype,
            device=self.device,
        )

        data_collection_complete = True
        for _ in range(self.cfg.n_initial_samples):
            t0 = time.time()
            pref = self.dirichlet.sample((1,)).reshape(1, self.instance.n_objs)
            obj = self.scalarizer.evaluate(pref).to(device=self.device, dtype=self.dtype)
            data_collection += time.time() - t0
            if data_collection > self.cfg.time_limit:
                data_collection_complete = False
                break
            time_dict["data_collection"].append(data_collection)
            prefs = torch.cat([prefs, pref])
            objs = torch.cat([objs, obj])

        if data_collection_complete:
            print("Initial data generation complete.")

        return prefs, objs

    def run(self):
        set_global_seed(self.cfg.rseed)
        time_dict = {}

        print(f"Using BO device: {self.device}")
        print(f"Using BO dtype: {self.dtype}")
        print(f"Using reference point: {self.instance.reference_point.tolist()}")

        surrogate = build_surrogate(
            self.cfg.surrogate, device=self.device, dtype=self.dtype
        )
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
            device=self.device,
            dtype=self.dtype,
        )
        print(
            f"Using acquisition: {self.cfg.acquisition.name} "
            f"| rseed: {self.cfg.rseed}"
        )

        prefs, objs = self.prepare_initial_training_data(time_dict)
        print(f"Starting BO loop for {self.cfg.n_iterations} iterations...")
        time_dict["iterations"] = []
        time_iterations = 0.0
        data_collection_time = (
            time_dict["data_collection"][-1] if time_dict["data_collection"] else 0.0
        )
        for _ in range(prefs.shape[0], self.cfg.n_iterations):
            t0 = time.time()

            model = surrogate.fit(prefs, objs, time_dict)
            new_prefs = acquisition.generate_candidates(model, prefs, time_dict).to(
                device=self.device, dtype=self.dtype
            )
            elapsed_without_eval = data_collection_time + time_iterations + (
                time.time() - t0
            )
            if elapsed_without_eval > float(self.cfg.time_limit):
                print("Time limit reached.")
                break

            new_objs = self.scalarizer.evaluate(new_prefs).to(
                device=self.device, dtype=self.dtype
            )
            time_iterations += time.time() - t0

            if data_collection_time + time_iterations > float(self.cfg.time_limit):
                print("Time limit reached.")
                break

            time_dict["iterations"].append(time_iterations)
            prefs = torch.cat([prefs, new_prefs])
            objs = torch.cat([objs, new_objs])

        prefs_np = prefs.detach().cpu().numpy()
        objs_np = objs.detach().cpu().numpy()

        ideal_point = self.instance.ideal_point
        ref_point = self.instance.reference_point
        t0 = time.time()
        iter_records = compute_iteration_stats(
            objs_np,
            ref_point,
            ideal_point=ideal_point,
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
            ref_point,
            ideal_point=ideal_point,
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
        self.save_result(prefs_np, objs_np, iter_records, final_record, time_dict)
        self.print_time_dict(time_dict)
        print("N evaluations (saved):", len(prefs_np))
        print("N evaluations (solver calls):", self.scalarizer.n_evaluations)
