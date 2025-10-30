from hydra import main as hydra_main

from ea.core.run import run_ea


def _build_mokp_problem(problem_cfg):
    from ea.problems.mokp import MOKP

    return MOKP(
        n_items=problem_cfg.n_items,
        n_objs=problem_cfg.n_objs,
        density=problem_cfg.density,
        iseed=problem_cfg.iseed,
    )


def build_problem(cfg):
    builders = {
        "mokp": _build_mokp_problem,
    }
    if cfg.problem.name not in builders:
        available = ", ".join(sorted(builders))
        raise ValueError(
            f"Unsupported problem '{cfg.problem.name}'. Supported problems: {available}"
        )
    return builders[cfg.problem.name](cfg.problem)


@hydra_main(config_path="configs", config_name="run_ea", version_base=None)
def main(cfg):
    problem = build_problem(cfg)
    run_ea(problem, cfg)


if __name__ == "__main__":
    main()
