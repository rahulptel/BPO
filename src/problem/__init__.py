from .moap import MOAPInstance
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
    if problem_cfg.name == "moap":
        return MOAPInstance(
            n_agents=problem_cfg.n_agents,
            n_objs=problem_cfg.n_objs,
            cost_min=problem_cfg.cost_min,
            cost_max=problem_cfg.cost_max,
            iseed=problem_cfg.iseed,
            env=env,
            optimizer=optimizer,
        )
    else:
        raise ValueError(f"Unknown problem name: {problem_cfg.name}")


__all__ = ["build_instance", "MOKPInstance", "MOAPInstance"]
