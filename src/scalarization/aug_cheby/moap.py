import numpy as np
import torch


class BaseAugChebyMOAPScalarizer:
    def __init__(self, instance, rho=1e-4, name="Base"):
        self.name = name
        self.instance = instance
        self.rho = float(rho)
        self.n_evaluations = 0

        self._ideal_point_min = self.instance.ideal_point
        self._log_scalarization_start()

    def _log_scalarization_start(self):
        print(
            f"Scalarization: Augmented Tchebycheff "
            f"(solver={self.name}, rho={self.rho})"
        )
        print(f"Computed Ideal Point (Minimization): {self._ideal_point_min}\n")

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
            objective_vector = self._solve_scalarized(pref_norm)
            results.append(objective_vector)

        results = np.array(results).reshape(batch_size, self.instance.n_objs)
        if orig_type is torch.Tensor:
            results = torch.tensor(results, dtype=torch.float64)

        return results

    def _solve_scalarized(self, pref):
        """
        Solves the problem in minimization form
        """
        raise NotImplementedError


class GurobiAugChebyMOAPScalarizer(BaseAugChebyMOAPScalarizer):
    def __init__(self, instance, env, rho=1e-4):
        super().__init__(instance, rho=rho, name="Gurobi")
        self.env = env
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "GurobiAugChebyMOAPScalarizer requires the gurobipy package"
            ) from exc

        self._gp = gp
        self._GRB = GRB

    def _build_model(self, pref):
        gp = self._gp
        GRB = self._GRB
        model = gp.Model(env=self.env)
        x = model.addMVar(
            shape=(self.instance.n_agents, self.instance.n_tasks),
            vtype=GRB.BINARY,
            name="x",
        )
        alpha = model.addMVar(
            shape=1,
            vtype=GRB.CONTINUOUS,
            lb=-GRB.INFINITY,
            name="alpha",
        )

        model.addConstr(x.sum(axis=1) == 1, name="assign_agents")
        model.addConstr(x.sum(axis=0) == 1, name="assign_tasks")

        achievements_delta = []
        for j in range(self.instance.n_objs):
            cost = (self.instance.costs[:, :, j] * x).sum()
            achievement_delta = cost - self._ideal_point_min[j]
            achievements_delta.append(achievement_delta)
            model.addConstr(alpha >= pref[j] * achievement_delta)

        augmentation = self.rho * gp.quicksum(
            pref[j] * achievements_delta[j] for j in range(self.instance.n_objs)
        )
        model.setObjective(alpha + augmentation, GRB.MINIMIZE)

        model._x = x
        return model

    def _get_solution(self, model):
        GRB = self._GRB
        if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            return model._x.X
        return None

    def _solve_scalarized(self, pref):
        self.n_evaluations += 1
        if isinstance(pref, torch.Tensor):
            pref = pref.detach().cpu().numpy()

        model = self._build_model(pref)
        model.optimize()
        solution = self._get_solution(model)
        if solution is None:
            raise RuntimeError(f"Gurobi solver failed for pref={pref.tolist()}")

        true_objective = np.sum(
            self.instance.costs * solution[:, :, None], axis=(0, 1)
        )
        return true_objective
