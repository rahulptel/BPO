import random
from pathlib import Path

import numpy as np
import torch
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)

SRC_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = SRC_DIR.parent / "outputs"


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dirichlet_initial_design(n_points, dim):
    base = torch.ones(dim, dtype=torch.get_default_dtype())
    distribution = torch.distributions.dirichlet.Dirichlet(base)
    return distribution.sample((n_points,)).reshape(n_points, dim)


def compute_hypervolume(Y_nd, ref_point, ideal_point=None, normalize=True):
    bd = FastNondominatedPartitioning(ref_point=ref_point, Y=Y_nd)
    hv_val = bd.compute_hypervolume().item()
    if normalize and ideal_point is not None:
        return normalize_hypervolume(hv_val, ideal_point)
    return hv_val


def normalize_hypervolume(unnorm_hv, ideal_point):
    if ideal_point is None:
        return unnorm_hv

    denom = np.abs(ideal_point).prod()
    if denom == 0:
        return unnorm_hv

    return unnorm_hv / denom
