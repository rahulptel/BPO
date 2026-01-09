import gurobipy as gp
import numpy as np
from gurobipy import GRB


class KSAProblem:
    def __init__(self, instance, env=None, objective_index=0, delta=1):
        self.instance = instance
        self.env = env
        self.delta = float(delta)
        self.K = int(objective_index)

        self.n_objectives = self.instance.n_objs
        if self.K < 0 or self.K >= self.n_objectives:
            raise ValueError(f"Invalid objective index: {self.K}")

        self.mask = np.ones(self.n_objectives, dtype=bool)
        self.mask[self.K] = False

        self.lb = np.asarray([])
        self.ub = np.asarray([])

        self.m = None
        self.c_epsilon = None
        self.c_obj_K_ub = None
        self.objectives_expr = []

        self._initialize_base_model()

    def _configure_model(self, model):
        model.setParam("OutputFlag", 0)
        model.setParam("MIPGap", 0)

    def _initialize_base_model(self):
        raise NotImplementedError

    def initialize_two_stage_model(self):
        raise NotImplementedError

    def _solution_vector(self):
        raise NotImplementedError

    def _objective_values(self, x):
        raise NotImplementedError

    def _bounds_from_instance(self):
        raise NotImplementedError

    def _update_epsilon_constraints(self, epsilon):
        for i in range(self.n_objectives - 1):
            self.c_epsilon[i].rhs = epsilon[i] - self.delta

    def _solve_p(self, epsilon):
        self.m.setObjective(self.objectives_expr[self.K], GRB.MINIMIZE)
        self._update_epsilon_constraints(epsilon)
        self.c_obj_K_ub.rhs = self.ub[self.K]
        self.m.update()
        self.m.optimize()

        if self.m.getAttr("Status") != GRB.OPTIMAL:
            return None, None
        z = float(self.m.ObjVal)
        x = self._solution_vector()
        return x, z

    def _solve_q(self, epsilon, z_star):
        self.m.setObjective(gp.quicksum(self.objectives_expr), GRB.MINIMIZE)
        self.c_obj_K_ub.rhs = z_star
        self.m.update()
        self.m.optimize()

        if self.m.getAttr("Status") != GRB.OPTIMAL:
            return None, None
        z = float(self.m.ObjVal)
        x = self._solution_vector()
        return x, z

    def get_bounds(self):
        lb, ub = self._bounds_from_instance()
        self.lb = np.asarray(lb, dtype=np.float64)
        self.ub = np.asarray(ub, dtype=np.float64)

        return self.lb, self.ub

    def get_solution(self, epsilon=None):
        x, z = None, None

        _, z_p = self._solve_p(epsilon)
        if z_p is not None:
            x_q, z_q = self._solve_q(epsilon, z_p)
            if z_q is not None:
                x = x_q
                z = self._objective_values(x)

        return x, z


class KSAMOKPProblem(KSAProblem):
    def __init__(self, instance, env=None, objective_index=0, delta=1):
        super().__init__(
            instance, env=env, objective_index=objective_index, delta=delta
        )
        self.n_variables = instance.n_items
        self.objectives = -instance.values.T
        self.weight = instance.weights
        self.capacity = instance.capacity

        self.v_items = None
        self.c_capacity = None
        self.objectives_bar = self.objectives[self.mask, :]

    def _initialize_base_model(self):
        self.m = gp.Model(env=self.env) if self.env is not None else gp.Model()
        self._configure_model(self.m)
        self.v_items = self.m.addMVar(
            self.n_variables, vtype=GRB.BINARY, name="item"
        )
        self.c_capacity = self.m.addConstr(
            self.weight @ self.v_items <= self.capacity, name="capacity"
        )
        self.objectives_expr = [
            self.objectives[i] @ self.v_items for i in range(self.n_objectives)
        ]
        self.m.update()

    def initialize_two_stage_model(self):
        self.c_epsilon = self.m.addConstrs(
            (
                self.objectives_bar[i] @ self.v_items <= self.ub[self.mask][i]
                for i in range(self.n_objectives - 1)
            ),
            name="epsilon",
        )
        self.c_obj_K_ub = self.m.addConstr(
            self.objectives[self.K, :] @ self.v_items <= self.ub[self.K],
            name="obj_k_ub",
        )
        self.m.update()

    def _solution_vector(self):
        return self.v_items.X

    def _objective_values(self, x):
        return self.objectives @ x

    def _bounds_from_instance(self):
        ideal = np.asarray(self.instance.ideal_point, dtype=np.float64)
        reference = np.asarray(self.instance.reference_point, dtype=np.float64)
        return -ideal, -reference


class KSAMOAPProblem(KSAProblem):
    def __init__(self, instance, env=None, objective_index=0, delta=1):
        self.n_agents = instance.n_agents
        self.n_tasks = instance.n_tasks
        self.costs = instance.costs.astype(np.float64)

        self.v_assign = None
        self.c_assign_agents = None
        self.c_assign_tasks = None
        self.objectives_bar_expr = None

        super().__init__(
            instance, env=env, objective_index=objective_index, delta=delta
        )
        self.objectives_bar_expr = [
            self.objectives_expr[i]
            for i in range(self.n_objectives)
            if i != self.K
        ]

    def _initialize_base_model(self):
        self.m = gp.Model(env=self.env) if self.env is not None else gp.Model()
        self._configure_model(self.m)
        self.v_assign = self.m.addMVar(
            (self.n_agents, self.n_tasks), vtype=GRB.BINARY, name="assign"
        )
        self.c_assign_agents = self.m.addConstr(
            self.v_assign.sum(axis=1) == 1, name="assign_agents"
        )
        self.c_assign_tasks = self.m.addConstr(
            self.v_assign.sum(axis=0) == 1, name="assign_tasks"
        )
        self.objectives_expr = [
            (self.costs[:, :, i] * self.v_assign).sum()
            for i in range(self.n_objectives)
        ]
        self.m.update()

    def initialize_two_stage_model(self):
        self.c_epsilon = self.m.addConstrs(
            (
                self.objectives_bar_expr[i] <= self.ub[self.mask][i]
                for i in range(self.n_objectives - 1)
            ),
            name="epsilon",
        )
        self.c_obj_K_ub = self.m.addConstr(
            self.objectives_expr[self.K] <= self.ub[self.K], name="obj_k_ub"
        )
        self.m.update()

    def _solution_vector(self):
        return self.v_assign.X

    def _objective_values(self, x):
        return (self.costs * x).sum(axis=(0, 1))

    def _bounds_from_instance(self):
        ideal = np.asarray(self.instance.ideal_point, dtype=np.float64)
        reference = np.asarray(self.instance.reference_point, dtype=np.float64)
        return ideal, reference
