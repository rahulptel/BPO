import numpy as np

import gurobipy as gp
from gurobipy import GRB


class KSAMOKPProblem:
    def __init__(self, instance, env=None, objective_index=0, delta=1):
        self.instance = instance
        self.env = env
        self.delta = float(delta)
        self.K = int(objective_index)

        self.n_objectives = self.instance.n_objs
        self.n_variables = self.instance.n_items
        if self.K < 0 or self.K >= self.n_objectives:
            raise ValueError(f"Invalid objective index: {self.K}")

        self.mask = np.ones(self.n_objectives, dtype=bool)
        self.mask[self.K] = False

        self.objectives = -self.instance.values.T
        self.objectives_bar = self.objectives[self.mask, :]
        self.weight = self.instance.weights
        self.capacity = self.instance.capacity

        self.lb = np.asarray([])
        self.ub = np.asarray([])

        self.m = None
        self.v_items = None
        self.c_capacity = None
        self.c_epsilon = None
        self.c_obj_K_ub = None
        self.objectives_expr = []

        self._initialize_base_model()

    def _configure_model(self, model):
        model.setParam("OutputFlag", 0)
        model.setParam("MIPGap", 0)

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
        self.c_epsilon = self.m.addConstr(
            self.objectives_bar @ self.v_items <= self.ub[self.mask]
        )
        self.c_obj_K_ub = self.m.addConstr(
            self.objectives[self.K, :] @ self.v_items <= self.ub[self.K]
        )
        self.m.update()

    def _solve_p(self, epsilon):
        self.m.setObjective(self.objectives_expr[self.K], GRB.MINIMIZE)
        for i in range(self.n_objectives - 1):
            self.c_epsilon[i].rhs = epsilon[i] - self.delta
        self.c_obj_K_ub.rhs = self.ub[self.K]
        self.m.update()
        self.m.optimize()

        if self.m.getAttr("Status") != GRB.OPTIMAL:
            return None, None
        z = int(self.m.ObjVal)
        x = self.v_items.X
        return x, z

    def _solve_q(self, epsilon, z_star):
        self.m.setObjective(gp.quicksum(self.objectives_expr), GRB.MINIMIZE)
        self.c_obj_K_ub.rhs = z_star
        self.m.update()
        self.m.optimize()

        if self.m.getAttr("Status") != GRB.OPTIMAL:
            return None, None
        z = int(self.m.ObjVal)
        x = self.v_items.X
        return x, z

    def get_bounds(self):
        lb, ub = [], []
        for obj_id in range(self.n_objectives):
            self.m.setObjective(self.objectives_expr[obj_id], GRB.MINIMIZE)
            self.m.update()
            self.m.optimize()
            if self.m.getAttr("Status") == GRB.INFEASIBLE:
                return None, ub
            lb.append(self.m.ObjVal)

            self.m.setObjective(self.objectives_expr[obj_id], GRB.MAXIMIZE)
            self.m.update()
            self.m.optimize()
            if self.m.getAttr("Status") == GRB.UNBOUNDED:
                ub.append(GRB.INFINITY)
            else:
                ub.append(int(self.m.ObjVal) + 1)

        self.lb = np.asarray(lb)
        self.ub = np.asarray(ub)

        return self.lb, self.ub

    def get_solution(self, epsilon=None):
        x, z = None, None

        x_p, z_p = self._solve_p(epsilon)
        if z_p is not None:
            x_q, z_q = self._solve_q(epsilon, z_p)
            if z_q is not None:
                x = x_q
                z = np.matmul(self.objectives, x)

        return x, z
