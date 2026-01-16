from pathlib import Path

import numpy as np
from pymoo.core.problem import Problem

SRC_DIR = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = SRC_DIR.parent.joinpath("outputs")


class PymooMOKPProblem(Problem):
    name = "mokp"
    encoding = "binary"

    def __init__(self, instance):
        self.instance = instance
        self._ideal_point = self.instance.ideal_point
        self._reference_point = self.instance.reference_point

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
        return OUTPUTS_DIR / "ea" / f"{str(self.instance)}" / f"rseed-{config.seed}"

    def default_ref_point(self):
        return self._reference_point.copy()

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

        out["F"] = profits
        out["G"] = constraint_violation.reshape(-1, 1)


class PymooMOAPProblem(Problem):
    name = "moap"
    encoding = "permutation"

    def __init__(self, instance):
        self.instance = instance
        self._ideal_point = self.instance.ideal_point
        self._reference_point = self.instance.reference_point
        self._n_agents = self.instance.n_agents
        self._n_tasks = self.instance.n_tasks
        if self._n_agents != self._n_tasks:
            raise ValueError(
                "Permutation encoding requires n_agents == n_tasks "
                f"(got {self._n_agents} agents, {self._n_tasks} tasks)."
            )

        super().__init__(
            n_var=self._n_agents,
            n_obj=self.instance.n_objs,
            n_ieq_constr=0,
            n_eq_constr=0,
            xl=0,
            xu=self._n_tasks - 1,
            vtype=int,
        )

    def metadata(self):
        return self.instance.metadata()

    def io_base_dir(self, config):
        return OUTPUTS_DIR / "ea" / f"{str(self.instance)}" / f"rseed-{config.seed}"

    def default_ref_point(self):
        return self._reference_point.copy()

    def ideal_point(self):
        return self._ideal_point.copy()

    @property
    def n_objs(self):
        return self.instance.n_objs

    def _evaluate(self, x, out, *args, **kwargs):
        permutations = np.atleast_2d(x).astype(np.int64)
        agent_idx = np.arange(self._n_agents)[None, :]
        costs = self.instance.costs[agent_idx, permutations].sum(axis=1)
        out["F"] = costs
