import numpy as np


class MOAPInstance:
    name = "moap"

    def __init__(
        self,
        n_agents=10,
        n_objs=3,
        cost_min=1,
        cost_max=20,
        iseed=123,
        env=None,
        optimizer="gurobi",
    ):
        self.n_agents = int(n_agents)
        self.n_tasks = int(n_agents)
        self.n_objs = int(n_objs)
        self.cost_min = int(cost_min)
        self.cost_max = int(cost_max)
        self.iseed = int(iseed)
        self.env = env

        rng = np.random.default_rng(self.iseed)
        self.costs = rng.integers(
            self.cost_min,
            self.cost_max + 1,
            size=(self.n_agents, self.n_tasks, self.n_objs),
        )

        if optimizer != "gurobi":
            raise ValueError(
                "MOAPInstance currently supports only the Gurobi optimizer."
            )
        self._ideal_point = None
        self._compute_ideal_point_fn = self._compute_ideal_point_gurobi

        print(f"MOAP Instance (iseed: {self.iseed}):")
        print(f"  Agents: {self.n_agents}, Tasks: {self.n_tasks}")
        print(f"  Objectives: {self.n_objs}")
        print(f"  Cost range: [{self.cost_min}, {self.cost_max}]\n")

    def __str__(self):
        return (
            f"moap-agents-{self.n_agents}_objs-{self.n_objs}_"
            f"iseed-{self.iseed}"
        )

    def metadata(self):
        return {
            "n_agents": self.n_agents,
            "n_tasks": self.n_tasks,
            "n_objs": self.n_objs,
            "cost_min": self.cost_min,
            "cost_max": self.cost_max,
            "iseed": self.iseed,
        }

    @property
    def reference_point(self):
        max_costs = self.costs.max(axis=(0, 1))
        return max_costs.astype(np.float64) * self.n_agents

    def _compute_ideal_point_gurobi(self):
        import gurobipy as gp
        from gurobipy import GRB

        ideal_point = np.zeros(self.n_objs, dtype=np.float64)
        for j in range(self.n_objs):
            with gp.Model(env=self.env) as model:
                x = model.addMVar(
                    shape=(self.n_agents, self.n_tasks),
                    vtype=GRB.BINARY,
                    name="x",
                )
                model.addConstr(x.sum(axis=1) == 1, name="assign_agents")
                model.addConstr(x.sum(axis=0) == 1, name="assign_tasks")

                costs = self.costs[:, :, j]
                model.setObjective((costs * x).sum(), GRB.MINIMIZE)
                model.optimize()

                if model.Status != GRB.OPTIMAL:
                    raise RuntimeError(f"Could not solve ideal point objective {j}")
                ideal_point[j] = model.ObjVal

        return ideal_point

    @property
    def ideal_point(self):
        if self._ideal_point is None:
            self._ideal_point = self._compute_ideal_point_fn()
        return self._ideal_point.copy()
