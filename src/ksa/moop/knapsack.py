import ast
from pathlib import Path

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from .problem import Problem

# Status codes
OPTIMAL = 2
INFEASIBLE = 3
INF_OR_UNBD = 4
UNBOUNDED = 5
NODE_LIMIT = 8
TIME_LIMIT = 9
INTERRUPTED = 11

# Set solver parameters
gp.setParam("OutputFlag", 0)
gp.setParam('MIPGap', 0)
gp.setParam('TimeLimit', 600)
gp.setParam('NodeLimit', 1000000000)
gp.setParam('Threads', 1)


class Knapsack(Problem):
    def __init__(self, filename):
        self.filename = filename
        self.delta = 1
        self.K = 0

        # Instance data
        self.mask = []
        self.n_objectives = 0
        self.n_variables = 0
        self.objectives = []
        self.objectives_bar = []
        self.weight = []
        self.capacity = 0

        # Bounds
        self.lb = np.asarray([])
        self.ub = np.asarray([])

        # Model
        self.m = None
        self.v_items = None
        self.c_capacity = None
        self.c_epsilon = None
        self.c_obj_K_ub = None
        self.objectives_expr = []

        self._load_data()
        self._initialize_base_model()

    def _load_data(self):
        blob = open(Path(self.filename), 'r')
        self.n_objectives = int(blob.readline().strip())
        self.mask = [True] * self.n_objectives
        self.mask[self.K] = False
        self.n_variables = int(blob.readline().strip())
        self.capacity = int(blob.readline().strip())

        _objectives = [blob.readline() for _ in range(self.n_objectives)]
        _objectives = "".join(_objectives).strip()
        self.objectives = -np.asarray(ast.literal_eval(_objectives))
        # Fetch coefficients of all objectives apart from Kth
        self.objectives_bar = self.objectives[self.mask, :]
        self.weight = np.asarray(ast.literal_eval(blob.readline().strip()))

    def _initialize_base_model(self):
        self.m = gp.Model('moo_knapsack')
        self.v_items = self.m.addMVar(self.n_variables, vtype=GRB.BINARY, name='item')
        self.c_capacity = self.m.addConstr(self.weight @ self.v_items <= self.capacity,
                                           name='capacity')
        self.objectives_expr = [self.objectives[i] @ self.v_items
                                for i in range(self.n_objectives)]
        self.m.update()

    def initialize_two_stage_model(self):
        self.c_epsilon = self.m.addConstr(self.objectives_bar @ self.v_items <= self.ub[self.mask])
        self.c_obj_K_ub = self.m.addConstr(self.objectives[self.K, :] @ self.v_items <= self.ub[self.K])
        self.m.update()

    def _solve_p(self, epsilon):
        self.m.setObjective(self.objectives_expr[self.K], GRB.MINIMIZE)
        # Add objective bounds
        # If delta is not subtracted, we can find solutions on the top-left,
        # top-right, or bottom-left corner of the rectangle R(ideal, epsilon).
        # In such cases, we will not be able to prune the space and algorithm
        # will not terminate.
        for i in range(self.n_objectives-1):
            self.c_epsilon[i].rhs = epsilon[i] - self.delta
        self.c_obj_K_ub[0].rhs = self.ub[self.K]
        self.m.update()
        self.m.optimize()

        if self.m.getAttr('Status') != OPTIMAL:
            return None, None
        z = int(self.m.ObjVal)
        x = self.v_items.X

        return x, z

    def _solve_q(self, epsilon, z_star):
        self.m.setObjective(sum(self.objectives_expr), GRB.MINIMIZE)
        self.c_obj_K_ub[0].rhs = z_star
        self.m.update()
        self.m.optimize()

        if self.m.getAttr('Status') != OPTIMAL:
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
            # Model infeasible
            if self.m.getAttr('status') == INFEASIBLE:
                return None, ub
            lb.append(self.m.ObjVal)

            self.m.setObjective(self.objectives_expr[obj_id], GRB.MAXIMIZE)
            self.m.update()
            self.m.optimize()
            if self.m.getAttr('status') == UNBOUNDED:
                ub.append(GRB.INFINITY)
            else:
                ub.append(int(self.m.ObjVal) + 1)

        self.lb = np.asarray(lb)
        self.ub = np.asarray(ub)

        return self.lb, self.ub

    def get_solution(self, epsilon=None):
        """Get a (efficient, nondominated) solution tuple for a
        given epsilon"""
        x, z = None, None

        x_p, z_p = self._solve_p(epsilon)
        if z_p is not None:
            x_q, z_q = self._solve_q(epsilon, z_p)
            if z_q is not None:
                assert (x_p == x_q).all()
                x = x_p
                z = np.matmul(self.objectives, x)

        return x, z
