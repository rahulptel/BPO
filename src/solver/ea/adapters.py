from pathlib import Path

import numpy as np
from pymoo.core.problem import Problem


SRC_DIR = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = SRC_DIR.parent.joinpath("outputs")


class PymooMOKPProblem(Problem):
    name = "mokp"

    def __init__(self, instance):
        self.instance = instance
        self._ideal_point = self.instance.ideal_point()

        super().__init__(
            n_var=self.instance.n_items,
            n_obj=self.instance.n_objs,
            n_ieq_constr=1,
            n_eq_constr=0,
            xl=0,
            xu=1,
            vtype=bool,
        )

    def metadata(self):
        return self.instance.metadata()

    def io_base_dir(self, config):
        return (
            OUTPUTS_DIR
            / "ea"
            / (
                f"{self.instance.base_descriptor()}"
                f"_seed-{config.seed}"
            )
        )

    def default_ref_point(self):
        return np.zeros(self.instance.n_objs, dtype=np.float64)

    def ideal_point(self):
        return self._ideal_point.copy()

    @property
    def n_objs(self):
        return self.instance.n_objs

    def _evaluate(self, x, out, *args, **kwargs):
        decisions = np.atleast_2d(x).astype(np.float64)

        profits = decisions @ self.instance.values
        weights = decisions @ self.instance.weights
        constraint_violation = weights - self.instance.capacity

        out["F"] = -profits
        out["G"] = constraint_violation.reshape(-1, 1)
