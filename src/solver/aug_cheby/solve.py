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
from omegaconf import OmegaConf

from scalarization.aug_cheby import AugChebyMOKPScalarizer
from utils import OUTPUTS_DIR


class AugChebySolver:
    def __init__(self, cfg, instance):
        self.cfg = cfg
        self.instance = instance
        self.scalarizer = AugChebyMOKPScalarizer(
            instance,
            rho=self.cfg.scalarization.rho,
        )

    @staticmethod
    def _set_global_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _sample_dirichlet(n_points, dim):
        base = torch.ones(dim, dtype=torch.get_default_dtype())
        distribution = torch.distributions.dirichlet.Dirichlet(base)
        return distribution.sample((n_points,)).reshape(n_points, dim)

    @staticmethod
    def _normalize_hypervolume(unnorm_hv, ideal_point):
        denom = torch.abs(ideal_point).prod().item()
        if denom == 0:
            return unnorm_hv
        return unnorm_hv / denom

    def _compute_iteration_stats(
        self,
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
                hv = self._normalize_hypervolume(hv_val, ideal_point)
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

    @staticmethod
    def _time_suffix(value):
        text = str(value)
        return text.replace(":", "-").replace("/", "-").replace(" ", "_")

    def _run_directory_chain(self):
        chain = [("rseed", self.cfg.rseed)]
        if self.cfg.n_iterations is not None:
            chain.append(("n_iter", self.cfg.n_iterations))
        if self.cfg.time_limit is not None:
            chain.append(("time", self._time_suffix(self.cfg.time_limit)))
        return chain

    def save_result(self, records, ref_point, time_dict):
        result = {
            "cfg": OmegaConf.to_container(self.cfg, resolve=True),
            "problem": self.instance.name,
            "problem_metadata": self.instance.metadata(),
            "scalarization": {
                "name": self.cfg.scalarization.name,
                "rho": self.cfg.scalarization.rho,
            },
            "n_evaluations": self.scalarizer.n_evaluations,
            "ref_point": ref_point.detach().cpu().tolist(),
            "iterations": records,
            "nondominated_solutions": records[-1]["n_nd"],
            "time_dict": time_dict,
        }

        output_dir = OUTPUTS_DIR / "aug_cheby" / self.instance.base_descriptor()
        for key, value in self._run_directory_chain():
            output_dir /= f"{key}-{value}"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = output_dir / f"run_aug_cheby_{timestamp}.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"Saved random solver results to {output_path}")
        return output_path

    def run(self):
        self._set_global_seed(self.cfg.rseed)

        ref_point = torch.tensor(
            self.instance.reference_point,
            dtype=torch.get_default_dtype(),
        )
        ideal_point = torch.tensor(
            self.instance.ideal_point,
            dtype=torch.get_default_dtype(),
        )

        print(f"Using reference point: {ref_point.tolist()}")

        time_dict = {"data_collection": 0.0, "iterations": 0.0}

        try:
            prefs = torch.empty((0, self.instance.n_objs), dtype=torch.get_default_dtype())
            objs = torch.empty((0, self.instance.n_objs), dtype=torch.get_default_dtype())

            all_prefs = []
            all_objs = []

            max_iterations = int(self.cfg.n_iterations)
            time_limit = float(self.cfg.time_limit)

            print(f"Starting random loop for {max_iterations} iterations...")

            for _ in range(max_iterations):
                if time_dict["iterations"] >= time_limit:
                    print("Time limit reached; stopping early.")
                    break

                t_iter_start = time.time()

                new_pref = self._sample_dirichlet(1, self.instance.n_objs)
                new_obj = self.scalarizer.evaluate(new_pref)

                prefs = torch.cat([prefs, new_pref])
                objs = torch.cat([objs, new_obj])

                all_prefs.append(prefs)
                all_objs.append(objs)

                time_dict["iterations"] += time.time() - t_iter_start

                if time_dict["iterations"] >= time_limit:
                    print("Time limit reached; stopping early.")
                    break

            if not all_prefs:
                print("No iterations executed; skipping result serialization.")
                return

            records = self._compute_iteration_stats(
                all_prefs,
                all_objs,
                ref_point,
                ideal_point,
                save_prefs=self.cfg.save_prefs,
                save_objs=self.cfg.save_objs,
            )

            print("N evaluations:", self.scalarizer.n_evaluations)

            self.save_result(records, ref_point, time_dict)
        finally:
            self.scalarizer.close()
