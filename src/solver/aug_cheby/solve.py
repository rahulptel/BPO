import json
import random
import time
from datetime import datetime
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf

from utils import OUTPUTS_DIR, compute_iteration_stats


class AugChebySolver:
    def __init__(self, cfg, instance, scalarizer):
        self.cfg = cfg
        self.instance = instance
        self.scalarizer = scalarizer

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
        return distribution.sample((n_points,)).reshape(n_points, dim).numpy()

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

    def save_result(self, prefs, objs, iter_records, final_record, time_dict):
        result = {
            "cfg": OmegaConf.to_container(self.cfg, resolve=True),
            "prefs": prefs.tolist(),
            "objs": objs.tolist(),
            "n_evaluations": len(objs),
            "iter_records": iter_records,
            "final_record": final_record,
            "time_dict": time_dict,
        }

        output_dir = OUTPUTS_DIR / "aug_cheby" / str(self.instance)
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
        print(f"Using reference point: {self.instance.reference_point}")

        time_dict = {"iterations": 0.0}
        prefs = np.empty((0, self.instance.n_objs))
        objs = np.empty((0, self.instance.n_objs))
        max_iterations = int(self.cfg.n_iterations)
        time_limit = float(self.cfg.time_limit)
        print(f"Starting random loop for {max_iterations} iterations...")

        for _ in range(max_iterations):
            if time_dict["iterations"] >= time_limit:
                print("Time limit reached; stopping early.")
                break

            t0 = time.time()

            new_pref = self._sample_dirichlet(1, self.instance.n_objs)
            new_obj = self.scalarizer.evaluate(new_pref)

            prefs = np.concatenate((prefs, new_pref), axis=0)
            objs = np.concatenate((objs, new_obj), axis=0)

            time_dict["iterations"] += time.time() - t0

        # Convert the objs to minimization form
        if self.scalarizer.maximize is True:
            objs = -objs

        t0 = time.time()
        iter_records = compute_iteration_stats(
            objs,
            self.instance.reference_point,
            ideal_point=self.instance.ideal_point,
            all_prefs=prefs,
            normalize_hypervolume=self.cfg.hypervolume.normalize,
            approx=self.cfg.hypervolume.approx,
            eps=self.cfg.hypervolume.eps,
            delta=self.cfg.hypervolume.delta,
            from_iteration=1,
            to_iteration=self.cfg.compute_stats_upto,
        )
        final_record = compute_iteration_stats(
            objs,
            self.instance.reference_point,
            ideal_point=self.instance.ideal_point,
            all_prefs=prefs,
            normalize_hypervolume=self.cfg.hypervolume.normalize,
            approx=self.cfg.hypervolume.approx,
            eps=self.cfg.hypervolume.eps,
            delta=self.cfg.hypervolume.delta,
            from_iteration=len(objs),
            to_iteration=len(objs),
        )
        time_dict["stats"] = time.time() - t0
        print("N evaluations:", self.scalarizer.n_evaluations)
        pprint(time_dict)
        self.save_result(prefs, objs, iter_records, final_record, time_dict)
