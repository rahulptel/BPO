import gurobipy as gp
import numpy as np
import torch
from gurobipy import GRB
from pyscipopt import Model, quicksum


class BaseAugChebyMOKPScalarizer:
    def __init__(self, instance, rho=1e-4, name="Base"):
        self.name = name
        self.instance = instance
        self.rho = float(rho)

        self.n_evaluations = 0

        # Assume: Ideal point provided considering maximization form
        self._ideal_point = self.instance.ideal_point
        self._ideal_point_min = -self._ideal_point
        self._log_scalarization_start()

    def _log_scalarization_start(self):
        print(
            f"Scalarization: Augmented Tchebycheff "
            f"(solver={self.name}, rho={self.rho})"
        )
        print(f"Computed Ideal Point: {self.ideal_point}\n")

    def evaluate(self, prefs):
        orig_type = None
        if isinstance(prefs, torch.Tensor):
            orig_type = torch.Tensor
            prefs = prefs.detach().cpu().numpy()
        else:
            prefs = np.asarray(prefs)

        batch_size, dim = prefs.shape
        assert dim == self.instance.n_objs

        results = []
        for pref in prefs:
            pref_norm = pref / np.sum(pref)
            objective_vector = self._solve_scalarized(pref_norm, maximize=True)
            results.append(objective_vector)

        results = np.array(results).reshape(batch_size, self.instance.n_objs)
        if orig_type is torch.Tensor:
            results = torch.tensor(results, dtype=torch.float64)

        return results

    @property
    def ideal_point(self):
        return self._ideal_point

    def _solve_scalarized(self, pref_vec, maximize=True):
        """
        Solves the problem in minimization form
        If maximize is True, we return the negative of objective value.
        """
        raise NotImplementedError


class GurobiAugChebyMOKPScalarizer(BaseAugChebyMOKPScalarizer):
    def __init__(self, instance, env, rho=1e-4):
        super().__init__(instance, rho=rho, name="Gurobi")
        self.env = env

    def _build_model(self, pref):
        model = gp.Model(env=self.env)
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
        for j in range(self.instance.n_objs):
            value = -(self.instance.values[:, j] @ x)
            achievements.append(value)
            achievements_delta.append(value - self._ideal_point_min[j])

        for j in range(self.instance.n_objs):
            model.addConstr(alpha >= pref[j] * achievements_delta[j])

        augmentation = self.rho * gp.quicksum(achievements_delta)
        model.setObjective(alpha + augmentation, GRB.MINIMIZE)

        model._x = x
        return model

    def _get_solution(self, model):
        if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            return model._x.X
        return None

    def _solve_scalarized(self, pref, maximize=True):
        self.n_evaluations += 1
        if isinstance(pref, torch.Tensor):
            pref = pref.detach().cpu().numpy()

        model = self._build_model(pref)
        model.optimize()
        solution = self._get_solution(model)
        if solution is None:
            raise RuntimeError(f"Gurobi solver failed for pref={pref.tolist()}")

        true_objective = self.instance.values.T @ solution
        return true_objective if maximize else -true_objective


class SCIPAugChebyMOKPScalarizer(BaseAugChebyMOKPScalarizer):
    def __init__(self, instance, rho=1e-4, threads=1, time_limit=100):
        super().__init__(instance, rho=rho, name="SCIP")
        self.threads = int(threads)
        self.time_limit = float(time_limit)

    def _build_model(self, pref):
        model = Model("aug_cheby_mokp_scip")
        model.setIntParam("display/verblevel", 0)
        if self.threads > 0:
            model.setIntParam("parallel/maxnthreads", self.threads)
        if self.time_limit > 0:
            model.setRealParam("limits/time", self.time_limit)

        x_vars = []
        for i in range(self.instance.n_items):
            var = model.addVar(name=f"x_{i}", vtype="BINARY")
            x_vars.append(var)
        alpha = model.addVar(
            name="alpha", vtype="CONTINUOUS", lb=-model.infinity(), ub=model.infinity()
        )

        capacity_expr = quicksum(
            self.instance.weights[i] * x_vars[i] for i in range(self.instance.n_items)
        )
        model.addCons(capacity_expr <= self.instance.capacity, name="capacity")

        achievements_delta = []
        for j in range(self.instance.n_objs):
            value_expr = -quicksum(
                self.instance.values[i, j] * x_vars[i]
                for i in range(self.instance.n_items)
            )
            achievement_delta = value_expr - self._ideal_point_min[j]
            achievements_delta.append(achievement_delta)
            model.addCons(alpha >= float(pref[j]) * achievement_delta)

        augmentation = self.rho * quicksum(achievements_delta)
        model.setObjective(alpha + augmentation, "minimize")

        model._x_vars = x_vars
        return model

    def _get_solution(self, model, pref):
        status = model.getStatus()
        if status not in {"optimal", "timelimit"}:
            raise RuntimeError(
                f"SCIP solver failed for pref={pref.tolist()} with status {status}"
            )

        sol = model.getBestSol()
        if sol is None:
            raise RuntimeError(
                f"SCIP did not return a solution for pref={pref.tolist()}"
            )

        return np.array([model.getSolVal(sol, var) for var in model._x_vars])

    def _solve_scalarized(self, pref, maximize=True):
        self.n_evaluations += 1
        if not isinstance(pref, np.ndarray):
            pref = pref.detach().cpu().numpy()

        model = self._build_model(pref)
        model.optimize()
        solution_x = self._get_solution(model, pref)

        true_objective = self.instance.values.T @ solution_x
        return true_objective if maximize else -true_objective


def build_scalarizer(cfg, instance, env=None):
    if cfg.scalarization.optimizer == "gurobi":
        return GurobiAugChebyMOKPScalarizer(instance, env, rho=cfg.scalarization.rho)
    elif cfg.scalarization.optimizer == "scip":
        return SCIPAugChebyMOKPScalarizer(
            instance, rho=cfg.scalarization.rho, time_limit=cfg.time_limit
        )
    else:
        raise ValueError("Invalid scalarizer")
