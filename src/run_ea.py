import torch
from hydra import main as hydra_main

from problem import build_instance
from solver.ea import EASolver
from utils import build_gurobi_env

torch.set_default_dtype(torch.float64)


def time_str_to_int(time_value):
    """Convert an HH:MM:SS-style string (or seconds) into an integer seconds value."""
    if isinstance(time_value, (int, float)):
        return int(time_value)
    time_parts = str(time_value).strip().split(":")
    if len(time_parts) != 3:
        raise ValueError(
            f"Expected time in 'HH:MM:SS' format, but received '{time_value}'."
        )
    hours, minutes, seconds = map(int, time_parts)
    return hours * 3600 + minutes * 60 + seconds


def build_pymoo_instance(problem_cfg, problem):
    if problem_cfg.name == "mokp":
        from solver.ea.adapters import PymooMOKPProblem

        return PymooMOKPProblem(problem)
    if problem_cfg.name == "moap":
        from solver.ea.adapters import PymooMOAPProblem

        return PymooMOAPProblem(problem)
    else:
        raise ValueError(f"Unknown problem name: {problem_cfg.name}")


@hydra_main(config_path="configs", config_name="run_ea", version_base=None)
def main(cfg):
    time_limit_seconds = time_str_to_int(cfg.algorithm.time)
    env = (
        build_gurobi_env(time_limit=time_limit_seconds)
        if cfg.optimizer == "gurobi"
        else None
    )
    if env is not None:
        env.start()

    for pid in range(cfg.from_pid, cfg.to_pid):
        cfg.problem.iseed = pid
        base_instance = build_instance(cfg.problem, env, cfg.optimizer)
        pymoo_instance = build_pymoo_instance(cfg.problem, base_instance)
        solver = EASolver(cfg, base_instance, pymoo_instance)
        solver.run()

    if env is not None:
        env.dispose()


if __name__ == "__main__":
    main()
