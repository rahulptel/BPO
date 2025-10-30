from pathlib import Path

import gurobipy as gp
import numpy as np
import torch
from gurobipy import GRB

from .base import Problem


class MOKP(Problem):
    name = "mokp"

    def __init__(self, n_items=50, n_objs=3, density=0.5, iseed=123, rho=1e-4):
        self.n_items = n_items
        self.n_objs = n_objs
        self.density = density
        self.iseed = iseed
        self.rho = rho
        self.n_evaluations = 0

        rng = np.random.default_rng(iseed)
        self.values = rng.integers(1, 1001, size=(n_items, n_objs))
        self.weights = rng.integers(1, 1001, size=n_items)
        self.capacity = int(np.sum(self.weights) * density)

        print(f"MOKP Instance (iseed: {iseed}):")
        print(f"  Items: {n_items}, Objectives: {n_objs}")
        print(f"  Knapsack Capacity: {self.capacity}")
        print(f"  Scalarization: Augmented Tchebycheff (rho={self.rho})\n")

        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.start()

        self.ideal_point_values = self._compute_ideal_point()
        self.ideal_point_min = -self.ideal_point_values
        print(f"Computed Ideal Point: {self.ideal_point_values}\n")

    def n_objectives(self):
        return self.n_objs

    def lambda_bounds(self):
        lower = torch.zeros(self.n_objs, dtype=torch.get_default_dtype())
        upper = torch.ones(self.n_objs, dtype=torch.get_default_dtype())
        return torch.stack([lower, upper])

    def lambda_equality_constraints(self):
        indices = torch.arange(self.n_objs)
        coeffs = torch.ones(self.n_objs, dtype=torch.get_default_dtype())
        return [(indices, coeffs, 1.0)]

    def default_ref_point(self):
        return torch.zeros(self.n_objs, dtype=torch.get_default_dtype())

    def ideal_point(self):
        return torch.tensor(self.ideal_point_values, dtype=torch.get_default_dtype())

    def initial_design(self, n):
        base = torch.ones(self.n_objs, dtype=torch.get_default_dtype())
        distribution = torch.distributions.dirichlet.Dirichlet(base)
        return distribution.sample((n,)).reshape(n, self.n_objs)

    def evaluate(self, lambda_batch):
        lambda_batch = lambda_batch.cpu()
        b, d = lambda_batch.shape
        assert d == self.n_objs

        results = []
        for i in range(b):
            lambda_vec = lambda_batch[i]
            lambda_vec_norm = lambda_vec / torch.sum(lambda_vec)
            obj_vec = self.solve_scalarized(lambda_vec_norm, maximize=True)
            results.append(obj_vec)

        results_tensor = torch.tensor(np.array(results), dtype=torch.float64)
        return results_tensor.reshape(b, self.n_objs)

    def metadata(self):
        return {
            "n_items": self.n_items,
            "n_objs": self.n_objs,
            "density": self.density,
            "rho": self.rho,
            "iseed": self.iseed,
        }

    def io_base_dir(self, config):
        return (
            Path("../outputs")
            / f"mokp-items-{self.n_items}_objs-{self.n_objs}_iseed-{self.iseed}_rseed-{config.rseed}"
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

    def solve_scalarized(self, lambda_vec, maximize=True):
        self.n_evaluations += 1
        if not isinstance(lambda_vec, np.ndarray):
            lambda_vec = lambda_vec.detach().cpu().numpy()

        with gp.Model(env=self.env) as m:
            x = m.addMVar(shape=self.n_items, vtype=GRB.BINARY, name="x")
            alpha = m.addMVar(
                shape=1, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="alpha"
            )

            m.addConstr(self.weights @ x <= self.capacity, name="capacity")

            y = []
            y_minus_z = []
            for j in range(self.n_objs):
                y_j = -(self.values[:, j] @ x)
                y.append(y_j)
                y_minus_z.append(y_j - self.ideal_point_min[j])

            for j in range(self.n_objs):
                m.addConstr(alpha >= lambda_vec[j] * y_minus_z[j], name=f"tcheby_{j}")

            augmentation_term = self.rho * gp.quicksum(y_minus_z)
            m.setObjective(alpha + augmentation_term, GRB.MINIMIZE)

            m.optimize()

            if m.Status == GRB.OPTIMAL:
                sol_x = x.X
                true_obj_vector = self.values.T @ sol_x
                return -true_obj_vector if not maximize else true_obj_vector

            print(
                f"Warning: Gurobi solver did not find an optimal solution for lambda={lambda_vec}"
            )
            return (
                np.array([1e6] * self.n_objs)
                if not maximize
                else np.array([0] * self.n_objs)
            )
