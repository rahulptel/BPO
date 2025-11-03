import torch
from hydra import main as hydra_main

from problem import build_instance
from solver.aug_cheby import AugChebySolver

torch.set_default_dtype(torch.float64)


@hydra_main(config_path="configs", config_name="run_aug_cheby", version_base=None)
def main(cfg):
    instance = build_instance(cfg.problem)
    solver = AugChebySolver(cfg, instance)
    solver.run()


if __name__ == "__main__":
    main()
