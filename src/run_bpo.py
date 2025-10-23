import argparse

import torch

from acquisition import available_acquisitions
from bpo.core.config import BOConfig
from bpo.core.run import run_bo
from bpo.problems import available_problems, build_problem

torch.set_default_dtype(torch.float64)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value_lower = value.lower()
    if value_lower in {"yes", "true", "t", "1"}:
        return True
    if value_lower in {"no", "false", "f", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preference-based Bayesian optimization runner."
    )
    parser.add_argument(
        "--problem",
        default="mokp",
        choices=available_problems(),
        help=f"Problem to optimize ({', '.join(available_problems())}).",
    )
    parser.add_argument(
        "--acquisition",
        default="qlogehvi",
        choices=available_acquisitions(),
        help=f"Acquisition function to use ({', '.join(available_acquisitions())}).",
    )
    parser.add_argument(
        "--rseed",
        type=int,
        default=123,
        help="Random seed for the BO run (controls samplers, Torch, etc.).",
    )
    parser.add_argument(
        "--iseed",
        type=int,
        default=123,
        help="Problem-specific seed (e.g., instance seed for MOKP).",
    )
    parser.add_argument(
        "--n-items",
        type=int,
        default=50,
        help="Number of items for the MOKP instance.",
    )
    parser.add_argument(
        "--n-objs",
        type=int,
        default=3,
        help="Number of objectives for the problem.",
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
        help="Capacity density fraction for the MOKP knapsack constraint.",
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
    return args


def main():
    args = parse_args()

    ref_point_tensor = (
        torch.tensor(args.ref_point, dtype=torch.get_default_dtype())
        if args.ref_point is not None
        else None
    )

    problem = build_problem(
        args.problem,
        n_items=args.n_items,
        n_objs=args.n_objs,
        density=args.density,
        iseed=args.iseed,
        rho=args.rho,
    )

    if ref_point_tensor is not None and ref_point_tensor.numel() != problem.n_objectives():
        raise ValueError(
            f"Ref point dimension {ref_point_tensor.numel()} does not match n_objs={problem.n_objectives()}."
        )

    config = BOConfig(
        acquisition=args.acquisition,
        rseed=args.rseed,
        n_initial_samples=args.n_initial_samples,
        n_iterations=args.n_iterations,
        mc_samples=args.mc_samples,
        batch_size_q=args.batch_size_q,
        num_restarts=args.num_restarts,
        raw_samples=args.raw_samples,
        should_maximize=args.should_maximize,
        sequential=args.sequential,
        ref_point=ref_point_tensor,
    )

    run_bo(problem, config)


if __name__ == "__main__":
    main()
