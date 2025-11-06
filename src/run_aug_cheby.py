import torch
from hydra import main as hydra_main

from problem import build_instance
from solver.aug_cheby import AugChebySolver
from utils import build_gurobi_env

torch.set_default_dtype(torch.float64)


@hydra_main(config_path="configs", config_name="run_aug_cheby", version_base=None)
def main(cfg):
    env = build_gurobi_env(time_limit=cfg.time_limit)
    env.start()
    for pid in range(cfg.from_pid, cfg.to_pid):
        cfg.problem.iseed = pid
        instance = build_instance(cfg.problem)
        solver = AugChebySolver(cfg, env, instance)
        solver.run()
    env.dispose()


if __name__ == "__main__":
    main()
