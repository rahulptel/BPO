import gurobipy as gp
import numpy as np
import torch
from gurobipy import GRB
from pyscipopt import Model, quicksum


def _log_scalarization_start(rho, solver_name, ideal_point):
    print(
        f"Scalarization: Augmented Tchebycheff "
        f"(solver={solver_name}, rho={rho})"
    )
    print(f"Computed Ideal Point: {ideal_point}\n")


class AugChebyMOKPScalarizer:
    def __init__(self, instance, rho=1e-4, threads=1, time_limit=100):
        self.instance = instance
        self.rho = float(rho)
        self.threads = int(threads)
        self.time_limit = float(time_limit)

        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.setParam("Threads", self.threads)
        self.env.setParam("TimeLimit", self.time_limit)
        self.env.start()

        self.n_evaluations = 0

        self._ideal_point = self.instance.ideal_point
        self._ideal_point_min = -self._ideal_point

        _log_scalarization_start(self.rho, "gurobi", self._ideal_point)

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

            if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
                solution_x = x.X
                true_objective = self.instance.values.T @ solution_x
                return true_objective if maximize else -true_objective

        raise RuntimeError(f"Gurobi solver failed for pref={pref_vec.tolist()}")

    def close(self):
        self.env.dispose()


class SCIPAugChebyMOKPScalarizer:
    def __init__(self, instance, rho=1e-4, threads=1, time_limit=100):
        self.instance = instance
        self.rho = float(rho)
        self.threads = int(threads)
        self.time_limit = float(time_limit)

        self.n_evaluations = 0

        self._ideal_point = self.instance.ideal_point
        self._ideal_point_min = -self._ideal_point

        _log_scalarization_start(self.rho, "scip", self._ideal_point)

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
        for j in range(self.n_objectives):
            value_expr = -quicksum(
                self.instance.values[i, j] * x_vars[i]
                for i in range(self.instance.n_items)
            )
            achievement_delta = value_expr - self._ideal_point_min[j]
            achievements_delta.append(achievement_delta)
            model.addCons(alpha >= float(pref_vec[j]) * achievement_delta)

        augmentation = self.rho * quicksum(achievements_delta)
        model.setObjective(alpha + augmentation, "minimize")

        model.optimize()

        status = model.getStatus()
        if status not in {"optimal", "timelimit"}:
            raise RuntimeError(
                f"SCIP solver failed for pref={pref_vec.tolist()} with status {status}"
            )

        sol = model.getBestSol()
        if sol is None:
            raise RuntimeError(
                f"SCIP did not return a solution for pref={pref_vec.tolist()}"
            )

        solution_x = np.array([model.getSolVal(sol, var) for var in x_vars])
        true_objective = self.instance.values.T @ solution_x
        return true_objective if maximize else -true_objective

    def close(self):
        pass


def build_aug_cheby_scalarizer(instance, scalarization_cfg, time_limit, threads=None):
    solver_name = getattr(scalarization_cfg, "optimizer", "scip")
    rho = getattr(scalarization_cfg, "rho", 1e-4)
    solver_key = str(solver_name).lower()
    thread_count = threads
    if thread_count is None:
        thread_count = getattr(scalarization_cfg, "threads", 1)
    thread_count = int(thread_count)

    if solver_key == "gurobi":
        scalarizer_cls = AugChebyMOKPScalarizer
    elif solver_key == "scip":
        scalarizer_cls = SCIPAugChebyMOKPScalarizer
    else:
        raise ValueError(
            f"Unknown AugCheby optimizer '{solver_name}'. "
            "Supported optimizers: 'scip', 'gurobi'."
        )

    return scalarizer_cls(
        instance,
        rho=rho,
        threads=thread_count,
        time_limit=time_limit,
    )
