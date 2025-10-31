import gurobipy as gp
import numpy as np
import torch
from gurobipy import GRB


class AugChebyMOKPScalarizer:
    def __init__(self, instance, rho=1e-4, threads=1):
        self.instance = instance
        self.rho = float(rho)
        self.threads = int(threads)

        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.setParam("Threads", self.threads)
        self.env.start()

        self.n_evaluations = 0

        self._ideal_point = self.instance.ideal_point
        self._ideal_point_min = -self._ideal_point

        print("Scalarization: Augmented Tchebycheff " f"(rho={self.rho})")
        print(f"Computed Ideal Point: {self._ideal_point}\n")

    @property
    def n_objectives(self):
        return self.instance.n_objs

    def evaluate(self, pref_batch):
        prefs = pref_batch.cpu()
        batch_size, dim = prefs.shape
        assert dim == self.n_objectives

        results = []
        for i in range(batch_size):
            pref = prefs[i]
            pref_norm = pref / torch.sum(pref)
            objective_vector = self._solve_scalarized(pref_norm, maximize=True)
            results.append(objective_vector)

        results_tensor = torch.tensor(np.array(results), dtype=torch.float64)
        return results_tensor.reshape(batch_size, self.n_objectives)

    def _solve_scalarized(self, pref_vec, maximize=True):
        self.n_evaluations += 1
        if not isinstance(pref_vec, np.ndarray):
            pref_vec = pref_vec.detach().cpu().numpy()

        with gp.Model(env=self.env) as model:
            x = model.addMVar(
                shape=self.instance.n_items,
                vtype=GRB.BINARY,
                name="x",
            )
            alpha = model.addMVar(
                shape=1,
                vtype=GRB.CONTINUOUS,
                lb=-GRB.INFINITY,
                name="alpha",
            )

            model.addConstr(
                self.instance.weights @ x <= self.instance.capacity,
                name="capacity",
            )

            achievements = []
            achievements_delta = []
            for j in range(self.n_objectives):
                value = -(self.instance.values[:, j] @ x)
                achievements.append(value)
                achievements_delta.append(value - self._ideal_point_min[j])

            for j in range(self.n_objectives):
                model.addConstr(alpha >= pref_vec[j] * achievements_delta[j])

            augmentation = self.rho * gp.quicksum(achievements_delta)
            model.setObjective(alpha + augmentation, GRB.MINIMIZE)

            model.optimize()

            if model.Status == GRB.OPTIMAL:
                solution_x = x.X
                true_objective = self.instance.values.T @ solution_x
                return true_objective if maximize else -true_objective

        raise RuntimeError(f"Gurobi solver failed for pref={pref_vec.tolist()}")

    def close(self):
        self.env.dispose()
