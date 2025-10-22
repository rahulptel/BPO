import argparse
import random
import time
from dataclasses import dataclass
from typing import Optional

import gurobipy as gp
import numpy as np
import torch

# BoTorch imports
from botorch import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gurobipy import GRB

from sampling_strategies import StrategyConfig, available_strategies, build_strategy

# Use float64 for better precision
torch.set_default_dtype(torch.float64)


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value_lower = value.lower()
    if value_lower in {"yes", "true", "t", "1"}:
        return True
    if value_lower in {"no", "false", "f", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'.")


@dataclass
class Config:
    sampling: str = "qlogehvi"
    seed: int = 123
    n_items: int = 50
    n_objs: int = 3
    n_initial_samples: int = 10
    n_iterations: int = 20
    mc_samples: int = 128
    batch_size_q: int = 2
    num_restarts: int = 10
    raw_samples: int = 512
    should_maximize: bool = True
    sequential: bool = True
    density: float = 0.5
    rho: float = 1e-4
    ref_point: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.ref_point is None:
            self.ref_point = torch.zeros(
                self.n_objs, dtype=torch.get_default_dtype()
            )
        else:
            if self.ref_point.numel() != self.n_objs:
                raise ValueError(
                    f"Ref point dimension {self.ref_point.numel()} does not match n_objs={self.n_objs}"
                )
            self.ref_point = self.ref_point.to(dtype=torch.get_default_dtype())


class MOKP:
    """
    A class to define and solve the Multi-Objective Knapsack Problem (MOKP).

    This version solves the (hard) subproblem using
    AUGMENTED TCHEBYCHEFF SCALARIZATION.

    This class serves as our "expensive black box" H(λ).
    It takes a preference vector λ and returns the true objective vector f(x*).
    """

    def __init__(self, n_items=50, n_objs=3, density=0.5, seed=123, rho=1e-4):
        self.n_items = n_items
        self.n_objs = n_objs
        self.rho = rho  # Augmentation parameter

        # Use a fixed seed for reproducibility
        rng = np.random.default_rng(seed)

        # Values: (n_items x n_objs) matrix
        self.values = rng.integers(1, 1001, size=(n_items, n_objs))

        # Weights: (n_items) vector
        self.weights = rng.integers(1, 1001, size=n_items)

        # Capacity: ~40% of total weight
        self.capacity = int(np.sum(self.weights) * density)

        print(f"MOKP Instance (Seed: {seed}):")
        print(f"  Items: {n_items}, Objectives: {n_objs}")
        print(f"  Knapsack Capacity: {self.capacity}")
        print(f"  Scalarization: Augmented Tchebycheff (rho={self.rho})\n")

        # Gurobi environment (suppress output)
        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.start()

        # --- NEW: Compute the Ideal Point z* ---
        # z*_j = min y_j(x) = min (-f_j(x)) = -max f_j(x)
        self.ideal_point = self._compute_ideal_point()
        self.ideal_point_min = -self.ideal_point
        print(f"Computed Ideal Point (for minimization): {self.ideal_point}\n")

    def _compute_ideal_point(self):
        """
        Solves 'n_objs' single-objective MOKPs to find the ideal point z*.
        """
        ideal_point = np.zeros(self.n_objs)
        for j in range(self.n_objs):
            with gp.Model(env=self.env) as m:
                x = m.addMVar(shape=self.n_items, vtype=GRB.BINARY, name="x")
                m.addConstr(self.weights @ x <= self.capacity, name="capacity")

                # Objective: max f_j(x) = max (values_j @ x)
                obj_coeffs = self.values[:, j]
                m.setObjective(obj_coeffs @ x, GRB.MAXIMIZE)

                m.optimize()

                if m.Status == GRB.OPTIMAL:
                    # We store the *negative* of the max value
                    ideal_point[j] = m.ObjVal
                else:
                    raise RuntimeError(f"Could not solve for ideal point obj {j}")
        return ideal_point

    def solve_scalarized(self, lambda_vec, maximize=False):
        """
        Solves the Augmented Tchebycheff scalarized MOKP.

        min  α + ρ * sum_j(y_j(x) - z*_j)
        s.t.
             α >= λ_j * (y_j(x) - z*_j)   for all j
             w^T x <= C
             x_i in {0, 1}

        where y_j(x) = -f_j(x) = -(values_j @ x)
        """

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

            # 4. Add Tchebycheff constraints:
            # alpha >= λ_j * (y_j(x) - z*_j)
            for j in range(self.n_objs):
                m.addConstr(alpha >= lambda_vec[j] * y_minus_z[j], name=f"tcheby_{j}")

            # 5. Define Objective Function:
            # min  α + ρ * sum_j(y_j(x) - z*_j)
            augmentation_term = self.rho * gp.quicksum(y_minus_z)
            m.setObjective(alpha + augmentation_term, GRB.MINIMIZE)

            # 6. Solve the model
            m.optimize()

            if m.Status == GRB.OPTIMAL:
                # Get the solution x*
                sol_x = x.X

                # Calculate the true (maximization) objective values
                true_obj_vector = self.values.T @ sol_x

                # Return the NEGATED objective vector for BoTorch (minimization)
                return -true_obj_vector if not maximize else true_obj_vector
            else:
                print(
                    f"Warning: Gurobi solver did not find an optimal solution for lambda={lambda_vec}"
                )
                # Return a very bad value (for minimization)
                return (
                    np.array([1e6] * self.n_objs)
                    if not maximize
                    else np.array([0] * self.n_objs)
                )

    def __call__(self, lambda_batch, maximize=False):
        """
        The black-box function H(λ) that BoTorch will call.
        (This function is identical to the previous version)
        """

        # Ensure input is on CPU for numpy/gurobi
        lambda_batch = lambda_batch.cpu()

        b, d = lambda_batch.shape
        assert d == self.n_objs

        results = []
        for i in range(lambda_batch.shape[0]):
            lambda_vec = lambda_batch[i]

            # Normalize lambda to sum to 1
            lambda_vec_norm = lambda_vec / torch.sum(lambda_vec)

            # Solve and get the (negated) objective vector
            obj_vec = self.solve_scalarized(lambda_vec_norm, maximize=maximize)
            results.append(obj_vec)

        # Reshape results back to (b, q, m) and return as a tensor
        results_tensor = torch.tensor(np.array(results), dtype=torch.float64)
        return results_tensor.reshape(b, self.n_objs)


def generate_random_samples(problem, n=10, maximize=False):
    """
    Generates n initial data points by sampling λ from a
    uniform Dirichlet distribution (i.e., uniform on the simplex).
    """
    print(f"Generating {n} initial data points...")
    # (n, d) tensor of λ vectors
    # We use q=1 for initial sampling
    train_x = (
        torch.distributions.dirichlet.Dirichlet(torch.ones(problem.n_objs))
        .sample((n,))
        .reshape(n, problem.n_objs)
    )

    # Evaluate the black box
    train_obj = problem(train_x, maximize=maximize)

    print("Initial data generation complete.")
    return train_x, train_obj


def initialize_model(train_x, train_obj):
    """
    Initializes a list of independent SingleTaskGPs, one for each
    (negated) objective.
    """
    # (b x q x d) -> (b*q, d)
    train_x_flat = train_x.reshape(-1, train_x.shape[-1])
    # (b x q x m) -> (b*q, m)
    train_obj_flat = train_obj.reshape(-1, train_obj.shape[-1])

    models = []
    for i in range(train_obj_flat.shape[-1]):
        # Get data for objective i
        train_y = train_obj_flat[:, i].unsqueeze(-1)

        # Use a Standardize transform for the outcomes
        models.append(
            SingleTaskGP(train_x_flat, train_y, outcome_transform=Standardize(m=1))
        )

    # Combine the models into a ModelListGP
    model = ModelListGP(*models)

    # Use a SumMarginalLogLikelihood for training
    mll = SumMarginalLogLikelihood(model.likelihood, model)

    return mll, model


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_sampling_strategy(config: Config):
    bounds = torch.stack(
        [torch.zeros(config.n_objs), torch.ones(config.n_objs)]
    )
    equality_constraints = [
        (
            torch.arange(config.n_objs),
            torch.ones(config.n_objs),
            1.0,
        )
    ]
    strategy_config = StrategyConfig(
        ref_point=config.ref_point,
        bounds=bounds,
        batch_size=config.batch_size_q,
        num_restarts=config.num_restarts,
        raw_samples=config.raw_samples,
        sequential=config.sequential,
        equality_constraints=equality_constraints,
        mc_samples=config.mc_samples,
        seed=config.seed,
    )
    return build_strategy(config.sampling, strategy_config)


def run_bo(config: Config):
    set_global_seed(config.seed)
    mokp = MOKP(
        n_items=config.n_items,
        n_objs=config.n_objs,
        density=config.density,
        seed=config.seed,
        rho=config.rho,
    )
    train_lambda, train_obj = generate_random_samples(
        mokp, n=config.n_initial_samples, maximize=config.should_maximize
    )

    sampling_strategy = build_sampling_strategy(config)
    print(
        f"Using sampling strategy: {sampling_strategy.__class__.__name__} | Seed: {config.seed}"
    )
    print(f"Starting BO loop for {config.n_iterations} iterations...")
    start_time = time.time()
    for i in range(config.n_iterations):
        # --- a. Fit the GP Surrogate Models ---
        mll, model = initialize_model(train_lambda, train_obj)
        fit_gpytorch_mll(mll)

        new_lambda = sampling_strategy.generate_candidates(
            model, train_lambda, train_obj
        )

        # --- d. Evaluate the Black Box ---
        new_obj = mokp(new_lambda, maximize=config.should_maximize)

        # --- e. Update the Dataset ---
        train_lambda = torch.cat([train_lambda, new_lambda])
        train_obj = torch.cat([train_obj, new_obj])

        pareto_mask = is_non_dominated(train_obj)
        bd = FastNondominatedPartitioning(ref_point=config.ref_point, Y=train_obj)
        volume = bd.compute_hypervolume().item() / torch.abs(
            torch.tensor(mokp.ideal_point.prod())
        )
        print(
            f"Iter {i+1}/{config.n_iterations} | ND: {pareto_mask.sum()} | Hypervolume: {volume:.4f}"
        )

    end_time = time.time()
    print(f"\nBO loop finished in {end_time - start_time:.2f} seconds.")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Preference-based Bayesian optimization runner."
    )
    parser.add_argument(
        "--sampling",
        default="qlogehvi",
        choices=available_strategies(),
        help=f"Sampling strategy to use ({', '.join(available_strategies())})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducible runs.",
    )
    parser.add_argument(
        "--n-items",
        type=int,
        default=50,
        help="Number of items in the knapsack instance.",
    )
    parser.add_argument(
        "--n-objs",
        type=int,
        default=3,
        help="Number of objectives for the knapsack problem.",
    )
    parser.add_argument(
        "--n-initial-samples",
        type=int,
        default=10,
        help="Number of initial design points sampled from the simplex.",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=20,
        help="Number of Bayesian optimization iterations to run.",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=128,
        help="Number of Monte Carlo samples used by the acquisition function.",
    )
    parser.add_argument(
        "--batch-size-q",
        type=int,
        default=2,
        help="Batch size (q) for candidate generation.",
    )
    parser.add_argument(
        "--num-restarts",
        type=int,
        default=10,
        help="Number of multistart restarts for acquisition optimization.",
    )
    parser.add_argument(
        "--raw-samples",
        type=int,
        default=512,
        help="Number of raw samples for acquisition optimization.",
    )
    parser.add_argument(
        "--should-maximize",
        type=str2bool,
        default=True,
        help="Whether to maximize the underlying objectives (true/false).",
    )
    parser.add_argument(
        "--sequential",
        type=str2bool,
        default=True,
        help="Whether to sample candidates sequentially (true/false).",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.5,
        help="Capacity density fraction for the knapsack constraint.",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=1e-4,
        help="Augmentation parameter for the scalarized MOKP.",
    )
    parser.add_argument(
        "--ref-point",
        type=float,
        nargs="+",
        default=None,
        help="Reference point for hypervolume (provide one value per objective).",
    )
    args = parser.parse_args()

    ref_point_tensor = (
        torch.tensor(args.ref_point, dtype=torch.get_default_dtype())
        if args.ref_point is not None
        else None
    )
    return Config(
        sampling=args.sampling,
        seed=args.seed,
        n_items=args.n_items,
        n_objs=args.n_objs,
        n_initial_samples=args.n_initial_samples,
        n_iterations=args.n_iterations,
        mc_samples=args.mc_samples,
        batch_size_q=args.batch_size_q,
        num_restarts=args.num_restarts,
        raw_samples=args.raw_samples,
        should_maximize=args.should_maximize,
        sequential=args.sequential,
        density=args.density,
        rho=args.rho,
        ref_point=ref_point_tensor,
    )


if __name__ == "__main__":
    config = parse_args()
    run_bo(config)
