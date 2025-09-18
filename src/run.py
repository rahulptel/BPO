import numpy as np
import torch
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from gpytorch.mlls import ExactMarginalLogLikelihood

from instance_generator import generate_instance
from solver import get_knapsack_model, solve_augmecon_subproblem


# --------------------------------------------------
# 2. Generate epsilon grid
# --------------------------------------------------
def generate_eps_grid(obj_mins, obj_maxs, num_points=5):
    eps2_vals = np.linspace(obj_mins[1], obj_maxs[1], num_points)
    eps3_vals = np.linspace(obj_mins[2], obj_maxs[2], num_points)
    xx, yy = np.meshgrid(eps2_vals, eps3_vals)
    return np.vstack([xx.ravel(), yy.ravel()]).T


# --------------------------------------------------
# 3. Example setup
# --------------------------------------------------
profits, weights, capacity = generate_instance(10, 3, 123)

# define bounds (can come from prior knowledge or initial random solves)
obj_mins = np.array([0.0, 0.0, 0.0])
obj_maxs = np.array([40.0, 40.0, 40.0])
eps_grid = generate_eps_grid(obj_mins, obj_maxs, num_points=5)  # (25, 2)

# initial training data (random subset of eps grid)
train_eps_indices = np.random.permutation(len(eps_grid))[:5]
train_eps = eps_grid[train_eps_indices]

model = get_knapsack_model(profits, weights, capacity)
train_obj = np.array(
    [solve_augmecon_subproblem(model.copy(), profits, eps) for eps in train_eps]
)

# --------------------------------------------------
# 4. Fit GP on eps -> objective mapping
# --------------------------------------------------
train_eps_torch = torch.from_numpy(train_eps).float()
train_obj_torch = torch.from_numpy(train_obj).float()

models = []
for i in range(train_obj_torch.shape[-1]):
    gp = SingleTaskGP(
        train_eps_torch, train_obj_torch[:, i : i + 1], outcome_transform=Standardize(1)
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    models.append(gp)
model = ModelListGP(*models)

# --------------------------------------------------
# 5. EHVI over eps-grid
# --------------------------------------------------
ref_point = train_obj_torch.min(dim=0).values - 1.0
partitioning = NondominatedPartitioning(ref_point=ref_point.tolist(), Y=train_obj_torch)

acq_func = qExpectedHypervolumeImprovement(
    model=model,
    ref_point=ref_point.tolist(),
    partitioning=partitioning,
)

# discrete candidate evaluation
eps_grid_torch = torch.from_numpy(eps_grid).float()
acq_vals = acq_func(eps_grid_torch.unsqueeze(1))  # Evaluate all candidates at once
best_idx = torch.argmax(acq_vals)
next_eps = eps_grid[best_idx]

print("Next epsilon grid point to evaluate:", next_eps)