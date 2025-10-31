import gurobipy as gp
import numpy as np
from gurobipy import GRB


class MOKPInstance:
    name = "mokp"

    def __init__(self, n_items=50, n_objs=3, density=0.5, iseed=123):
        self.n_items = int(n_items)
        self.n_objs = int(n_objs)
        self.density = float(density)
        self.iseed = int(iseed)

        rng = np.random.default_rng(self.iseed)
        self.values = rng.integers(1, 1001, size=(self.n_items, self.n_objs))
        self.weights = rng.integers(1, 1001, size=self.n_items)
        self.capacity = int(np.sum(self.weights) * self.density)

        self._ideal_point = None

        print(f"MOKP Instance (iseed: {self.iseed}):")
        print(f"  Items: {self.n_items}, Objectives: {self.n_objs}")
        print(f"  Knapsack Capacity: {self.capacity}\n")

    def __str__(self):
        return f"mokp-items-{self.n_items}_objs-{self.n_objs}_iseed-{self.iseed}"

    def metadata(self):
        return {
            "n_items": self.n_items,
            "n_objs": self.n_objs,
            "density": self.density,
            "iseed": self.iseed,
        }

    @property
    def reference_point(self):
        return np.zeros(self.n_objs, dtype=np.float64)

    def _compute_ideal_point(self):
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", 1)
        env.start()

        ideal_point = np.zeros(self.n_objs, dtype=np.float64)
        try:
            for j in range(self.n_objs):
                with gp.Model(env=env) as model:
                    x = model.addMVar(
                        shape=self.n_items,
                        vtype=GRB.BINARY,
                        name="x",
                    )
                    model.addConstr(self.weights @ x <= self.capacity, name="capacity")

                    coefficients = self.values[:, j]
                    model.setObjective(coefficients @ x, GRB.MAXIMIZE)

                    model.optimize()

                    if model.Status != GRB.OPTIMAL:
                        raise RuntimeError(
                            f"Could not solve for ideal point objective {j}"
                        )
                    ideal_point[j] = model.ObjVal
        finally:
            env.dispose()

        return ideal_point

    @property
    def ideal_point(self):
        if self._ideal_point is None:
            self._ideal_point = self._compute_ideal_point()
        return self._ideal_point.copy()
