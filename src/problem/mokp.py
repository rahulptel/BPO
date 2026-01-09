import numpy as np


class MOKPInstance:
    name = "mokp"

    def __init__(
        self, n_items=50, n_objs=3, density=0.5, iseed=123, env=None, optimizer="gurobi"
    ):
        self.n_items = int(n_items)
        self.n_objs = int(n_objs)
        self.density = float(density)
        self.iseed = int(iseed)
        self.env = env

        rng = np.random.default_rng(self.iseed)
        self.values = rng.integers(1, 1001, size=(self.n_items, self.n_objs))
        self.weights = rng.integers(1, 1001, size=self.n_items)
        self.capacity = int(np.sum(self.weights) * self.density)

        self._ideal_point = None
        self._compute_ideal_point_fn = (
            self._compute_ideal_point_gurobi
            if optimizer == "gurobi"
            else self._compute_ideal_point_scip
        )

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
        return self.values.min(axis=0) - 1.0        

    def _compute_ideal_point_gurobi(self):
        import gurobipy as gp
        from gurobipy import GRB

        ideal_point = np.zeros(self.n_objs, dtype=np.float64)
        for j in range(self.n_objs):
            with gp.Model(env=self.env) as model:
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
                    raise RuntimeError(f"Could not solve for ideal point objective {j}")
                ideal_point[j] = model.ObjVal

        return ideal_point

    def _compute_ideal_point_scip(self):
        from pyscipopt import Model, quicksum

        ideal_point = np.zeros(self.n_objs, dtype=np.float64)
        for j in range(self.n_objs):
            model = Model("mokp_ideal_point")
            model.setIntParam("display/verblevel", 0)

            x_vars = [
                model.addVar(name=f"x_{item_idx}", vtype="BINARY")
                for item_idx in range(self.n_items)
            ]

            capacity_expr = quicksum(
                self.weights[i] * x_vars[i] for i in range(self.n_items)
            )
            model.addCons(capacity_expr <= self.capacity, name="capacity")

            objective_expr = quicksum(
                float(self.values[i, j]) * x_vars[i] for i in range(self.n_items)
            )
            model.setObjective(objective_expr, sense="maximize")

            model.optimize()
            status = model.getStatus()
            if status != "optimal":
                raise RuntimeError(
                    f"Could not solve for ideal point objective {j} with status {status}"
                )

            sol = model.getBestSol()
            if sol is None:
                raise RuntimeError(
                    f"SCIP did not return a solution for ideal point objective {j}"
                )

            ideal_point[j] = sum(
                float(self.values[i, j]) * model.getSolVal(sol, x_vars[i])
                for i in range(self.n_items)
            )

        return ideal_point

    @property
    def ideal_point(self):
        if self._ideal_point is None:
            self._ideal_point = self._compute_ideal_point_fn()
        return self._ideal_point.copy()
