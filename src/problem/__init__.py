from .mokp import MOKPInstance


def build_instance(problem_cfg, env, optimizer):
    if problem_cfg.name == "mokp":
        return MOKPInstance(
            n_items=problem_cfg.n_items,
            n_objs=problem_cfg.n_objs,
            density=problem_cfg.density,
            iseed=problem_cfg.iseed,
            env=env,
            optimizer=optimizer,
        )
    else:
        raise ValueError(f"Unknown problem name: {problem_cfg.name}")


__all__ = ["build_instance", "MOKPInstance"]
