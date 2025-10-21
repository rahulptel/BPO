import argparse
import random
import time

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


class MOKPInstance:
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


def build_sampling_strategy(name: str, seed: int):
    bounds = torch.stack([torch.zeros(N_OBJS), torch.ones(N_OBJS)])
    equality_constraints = [(torch.arange(N_OBJS), torch.ones(N_OBJS), 1.0)]
    config = StrategyConfig(
        ref_point=REF_POINT,
        bounds=bounds,
        batch_size=BATCH_SIZE_Q,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        sequential=SEQUENTIAL,
        equality_constraints=equality_constraints,
        mc_samples=MC_SAMPLES,
        seed=seed,
    )
    return build_strategy(name, config)


def run_bo(sampling_name: str, seed: int):
    set_global_seed(seed)
    mokp_problem = MOKPInstance(n_items=N_ITEMS, n_objs=N_OBJS, seed=seed)
    train_lambda, train_obj = generate_random_samples(
        mokp_problem, n=N_INITIAL_SAMPLES, maximize=SHOULD_MAXIMIZE
    )

    sampling_strategy = build_sampling_strategy(sampling_name, seed)
    print(
        f"Using sampling strategy: {sampling_strategy.__class__.__name__} | Seed: {seed}"
    )
    print(f"Starting BO loop for {N_ITERATIONS} iterations...")
    start_time = time.time()
    for i in range(N_ITERATIONS):
        # --- a. Fit the GP Surrogate Models ---
        mll, model = initialize_model(train_lambda, train_obj)
        fit_gpytorch_mll(mll)

        new_lambda = sampling_strategy.generate_candidates(
            model, train_lambda, train_obj
        )

        # --- d. Evaluate the Black Box ---
        new_obj = mokp_problem(new_lambda, maximize=SHOULD_MAXIMIZE)

        # --- e. Update the Dataset ---
        train_lambda = torch.cat([train_lambda, new_lambda])
        train_obj = torch.cat([train_obj, new_obj])

        pareto_mask = is_non_dominated(train_obj)
        bd = FastNondominatedPartitioning(ref_point=REF_POINT, Y=train_obj)
        volume = bd.compute_hypervolume().item() / torch.abs(
            torch.tensor(mokp_problem.ideal_point.prod())
        )
        print(
            f"Iter {i+1}/{N_ITERATIONS} | ND: {pareto_mask.sum()} | Hypervolume: {volume:.4f}"
        )

    end_time = time.time()
    print(f"\nBO loop finished in {end_time - start_time:.2f} seconds.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preference-based Bayesian optimization runner."
    )
    parser.add_argument(
        "--sampling",
        default="qlogehvi",
        help=f"Sampling strategy to use ({', '.join(available_strategies())})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducible runs.",
    )
    return parser.parse_args()


# Problem & BO Parameters
N_ITEMS = 50
N_OBJS = 3
N_INITIAL_SAMPLES = 10  # Warm-up points
N_ITERATIONS = 20  # BO loop iterations
MC_SAMPLES = 128  # Samples for qEHVI
BATCH_SIZE_Q = 2  # q=1 for sequential optimization
NUM_RESTARTS = 10
RAW_SAMPLES = 512
REF_POINT = torch.zeros(N_OBJS)  # Reference point for hypervolume
SHOULD_MAXIMIZE = True  # Whether to maximize the objectives
SEQUENTIAL = True
if __name__ == "__main__":
    args = parse_args()
    run_bo(args.sampling, args.seed)

    # from botorch.test_functions.multi_objective import BraninCurrin
    # problem = BraninCurrin(negate=True)
    # print(problem.ref_point)
