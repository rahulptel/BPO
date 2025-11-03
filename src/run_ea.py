import torch
from hydra import main as hydra_main

from problem import build_instance
from solver.ea import EASolver

torch.set_default_dtype(torch.float64)


def build_pymoo_instance(problem_cfg, problem):
    if problem_cfg.name == "mokp":
        from solver.ea.adapters import PymooMOKPProblem

        return PymooMOKPProblem(problem)
    else:
        raise ValueError(f"Unknown problem name: {problem_cfg.name}")


@hydra_main(config_path="configs", config_name="run_ea", version_base=None)
def main(cfg):
    base_instance = build_instance(cfg.problem)
    pymoo_instance = build_pymoo_instance(cfg.problem, base_instance)
    solver = EASolver(cfg, base_instance, pymoo_instance)
    solver.run()


if __name__ == "__main__":
    main()
