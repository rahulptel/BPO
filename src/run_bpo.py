import torch
from hydra import main as hydra_main

from bpo.core.run import run_bo
from bpo.problems import available_problems

torch.set_default_dtype(torch.float64)


def _build_mokp_problem(problem_cfg):
    from bpo.problems.mokp import MOKP

    return MOKP(
        n_items=problem_cfg.n_items,
        n_objs=problem_cfg.n_objs,
        density=problem_cfg.density,
        iseed=problem_cfg.iseed,
        rho=problem_cfg.rho,
    )


def build_problem(cfg):
    builders = {
        "mokp": _build_mokp_problem,
    }

    if cfg.problem.name.lower() not in available_problems():
        available = ", ".join(sorted(builders))
        raise ValueError(
            f"Unsupported problem '{cfg.problem.name}'. Supported problems: {available}"
        )

    return builders[cfg.problem.name](cfg.problem)


@hydra_main(config_path="configs", config_name="run_bpo", version_base=None)
def main(cfg):
    problem = build_problem(cfg)
    run_bo(problem, cfg)


if __name__ == "__main__":
    main()
