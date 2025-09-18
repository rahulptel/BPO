import gurobipy as gp
import numpy as np
from gurobipy import GRB


def get_knapsack_model(profits, weights, capacity):
    """
    Generates the base Gurobi model for the knapsack problem.
    """
    n_items, n_objs = profits.shape
    m = gp.Model()
    m.Params.LogToConsole = 0

    # variables
    x = m.addVars(n_items, vtype=GRB.BINARY, name="x")
    s = m.addVars(n_objs - 1, vtype=GRB.CONTINUOUS, lb=0, name="s")

    # constraints
    m.addConstr(
        gp.quicksum(weights[i] * x[i] for i in range(n_items)) <= capacity,
        name="capacity",
    )
    # Epsilon constraints are added with a placeholder RHS.
    # The expression is sum(profit) + slack >= value.
    cons = [
        m.addConstr(
            gp.quicksum(profits[i, j + 1] * x[i] for i in range(n_items)) - s[j] == 0,
            name=f"eps{j}",
        )
        for j in range(n_objs - 1)
    ]

    m.update()
    m._x, m._s = x, s
    m._eps_cons = cons
    return m


def solve_augmecon_subproblem(model, profits, eps_point, obj_range=None, rho=1e5):
    """
    Solve triobjective knapsack at a given eps_point = (eps2, eps3).
    Objective 1 is primary, eps constraints on 2 and 3.
    """
    n_items, n_objs = profits.shape

    # Get variables and constraints from the model
    x_vars, s_vars, eps_cons = model._x, model._s, model._eps_cons

    # Set the RHS of epsilon constraints
    for j, cons in enumerate(eps_cons):
        cons.rhs = eps_point[j]

    # objective = maximize c1 + penalize slacks
    if obj_range is None:
        obj_range = [100 for _ in range(n_objs - 1)]

    model.setObjective(
        gp.quicksum(profits[i, 0] * x_vars[i] for i in range(n_items))
        + rho * gp.quicksum(s_vars[j] / obj_range[j] for j in range(n_objs - 1)),
        GRB.MAXIMIZE,
    )

    model.optimize()

    if model.Status != GRB.OPTIMAL:
        return None

    sol_x = np.array([x_vars[i].X for i in range(n_items)]).reshape(-1, 1)
    return np.matmul(np.array(profits).T, sol_x).reshape(-1).tolist()


def solve_single_objective_knapsack(model, profits, obj_index):
    """
    Solves the single-objective knapsack problem for a given objective.
    """
    n_items, n_objs = profits.shape
    x_vars, eps_cons = model._x, model._eps_cons

    # Deactivate epsilon constraints by setting RHS to -infinity
    for cons in eps_cons:
        cons.rhs = 0

    # Set objective for single objective optimization
    model.setObjective(
        gp.quicksum(profits[i, obj_index] * x_vars[i] for i in range(n_items)),
        GRB.MAXIMIZE,
    )

    model.optimize()

    if model.Status != GRB.OPTIMAL:
        return 0.0
    return model.ObjVal
