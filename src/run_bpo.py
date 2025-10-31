import torch
from hydra import main as hydra_main

from problem import MOKPInstance
from solver.bpo import BPOSolver

torch.set_default_dtype(torch.float64)


def build_instance(problem_cfg):
    if problem_cfg.name == "mokp":
        return MOKPInstance(
            n_items=problem_cfg.n_items,
            n_objs=problem_cfg.n_objs,
            density=problem_cfg.density,
            iseed=problem_cfg.iseed,
        )
    else:
        raise ValueError(f"Unknown problem name: {problem_cfg.name}")


@hydra_main(config_path="configs", config_name="run_bpo", version_base=None)
def main(cfg):
    instance = build_instance(cfg.problem)
    solver = BPOSolver(cfg, instance)
    solver.run()


if __name__ == "__main__":
    main()
