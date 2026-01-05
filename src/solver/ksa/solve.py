import json
import time
from datetime import datetime

import numpy as np
from omegaconf import OmegaConf

from utils import OUTPUTS_DIR, compute_iteration_stats

from .adapters import KSAMOKPProblem
from .epsilon_search import EpsilonSearch


class KSASolver:
    def __init__(self, cfg, instance, env=None):
        self.cfg = cfg
        self.instance = instance
        self.env = env

    @staticmethod
    def _time_suffix(value):
        text = str(value)
        return text.replace(":", "-").replace("/", "-").replace(" ", "_")

    def _run_directory_chain(self):
        chain = [("obj", self.cfg.objective_index), ("delta", self.cfg.delta)]
        if self.cfg.time_limit is not None:
            chain.append(("time", self._time_suffix(self.cfg.time_limit)))
        return chain

    def save_result(
        self,
        x_sol,
        y_sol,
        iter_records,
        final_record,
        time_dict,
        timer,
        n_evaluations,
    ):
        result = {
            "cfg": OmegaConf.to_container(self.cfg, resolve=True),
            "x_sol": x_sol,
            "y_sol": y_sol,
            "n_evaluations": n_evaluations,
            "iter_records": iter_records,
            "final_record": final_record,
            "time_dict": time_dict,
            "timer": timer,
        }

        output_dir = OUTPUTS_DIR / "ksa" / str(self.instance)
        for key, value in self._run_directory_chain():
            output_dir /= f"{key}-{value}"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = output_dir / f"run_ksa_{timestamp}.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"Saved KSA results to {output_path}")
        return output_path

    def run(self):
        t0 = time.time()
        problem = KSAMOKPProblem(
            self.instance,
            env=self.env,
            objective_index=self.cfg.objective_index,
            delta=self.cfg.delta,
        )
        search = EpsilonSearch(problem)
        search.run()
        total_time = time.time() - t0
        time_dict = {"search": total_time}

        if search.Z_n:
            objs = np.array(list(search.Z_n))
        else:
            objs = np.empty((0, self.instance.n_objs))
        xs = np.array(list(search.X_e)) if search.X_e else None
        n_evaluations = search.n_evaluations
        timer = list(search.timer)

        iter_records, final_record = [], []
        if objs.size > 0:
            iter_records = compute_iteration_stats(
                objs,
                self.instance.reference_point,
                ideal_point=self.instance.ideal_point,
                normalize_hypervolume=self.cfg.hypervolume.normalize,
                approx=self.cfg.hypervolume.approx,
                eps=self.cfg.hypervolume.eps,
                delta=self.cfg.hypervolume.delta,
                lib=self.cfg.hypervolume.lib,
                from_iteration=1,
                to_iteration=self.cfg.compute_stats_upto,
            )
            final_record = compute_iteration_stats(
                objs,
                self.instance.reference_point,
                ideal_point=self.instance.ideal_point,
                normalize_hypervolume=self.cfg.hypervolume.normalize,
                approx=self.cfg.hypervolume.approx,
                eps=self.cfg.hypervolume.eps,
                delta=self.cfg.hypervolume.delta,
                lib=self.cfg.hypervolume.lib,
                from_iteration=len(objs),
                to_iteration=len(objs),
            )

        x_sol = xs.tolist() if xs is not None else None
        y_sol = objs.tolist() if objs.size > 0 else []
        self.save_result(
            x_sol,
            y_sol,
            iter_records,
            final_record,
            time_dict,
            timer,
            n_evaluations,
        )
