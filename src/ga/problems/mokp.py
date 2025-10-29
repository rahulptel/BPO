from pathlib import Path

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from pymoo.core.problem import Problem

SRC_DIR = Path(__file__).parent.parent.parent
OUTPUTS_DIR = SRC_DIR.parent.joinpath("outputs")


class MOKP(Problem):
    name = "mokp"

    def __init__(self, n_items=50, n_objs=3, density=0.5, iseed=123):
        self.n_items = n_items
        self.n_objs = n_objs
        self.density = density
        self.iseed = iseed

        rng = np.random.default_rng(iseed)
        self.values = rng.integers(1, 1001, size=(n_items, n_objs))
        self.weights = rng.integers(1, 1001, size=n_items)
        self.capacity = int(np.sum(self.weights) * density)

        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.start()

        self.ideal_point_values = self._compute_ideal_point()

        super().__init__(
            n_var=n_items,
            n_obj=n_objs,
            n_ieq_constr=1,
            n_eq_constr=0,
            xl=0,
            xu=1,
            vtype=bool,
        )

    def _compute_ideal_point(self):
        ideal_point = np.zeros(self.n_objs)
        for j in range(self.n_objs):
            with gp.Model(env=self.env) as m:
                x = m.addMVar(shape=self.n_items, vtype=GRB.BINARY, name="x")
                m.addConstr(self.weights @ x <= self.capacity, name="capacity")

                obj_coeffs = self.values[:, j]
                m.setObjective(obj_coeffs @ x, GRB.MAXIMIZE)

                m.optimize()

                if m.Status == GRB.OPTIMAL:
                    ideal_point[j] = m.ObjVal
                else:
                    raise RuntimeError(f"Could not solve for ideal point obj {j}")
        return ideal_point

    def metadata(self):
        return {
            "n_items": self.n_items,
            "n_objs": self.n_objs,
            "density": self.density,
            "iseed": self.iseed,
        }

    def io_base_dir(self, config):
        return (
            OUTPUTS_DIR
            / "ga"
            / f"mokp-items-{self.n_items}_objs-{self.n_objs}_iseed-{self.iseed}_seed-{config.seed}"
        )

    def default_ref_point(self):
        return np.zeros(self.n_objs, dtype=np.float64)

    def ideal_point(self):
        return self.ideal_point_values.copy()

    def _evaluate(self, x, out, *args, **kwargs):
        decisions = np.atleast_2d(x).astype(np.float64)

        profits = decisions @ self.values
        weights = decisions @ self.weights
        constraint_violation = weights - self.capacity

        out["F"] = -profits
        out["G"] = constraint_violation.reshape(-1, 1)
