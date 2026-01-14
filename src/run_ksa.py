import time

import torch
from hydra import main as hydra_main

from problem import build_instance
from solver.ksa import KSASolver
from utils import build_gurobi_env

torch.set_default_dtype(torch.float64)


@hydra_main(config_path="configs", config_name="run_ksa", version_base=None)
def main(cfg):
    if cfg.optimizer != "gurobi":
        raise ValueError("KSA currently supports only the Gurobi optimizer.")

    t0 = time.time()
    env = build_gurobi_env(
        time_limit=cfg.time_limit, output_flag=cfg.outputflag, mipgap=cfg.mipgap
    )
    env.start()
    print("Time taken to start gurobi env: ", time.time() - t0)

    for pid in range(cfg.from_pid, cfg.to_pid):
        cfg.problem.iseed = pid
        instance = build_instance(cfg.problem, env, cfg.optimizer)
        solver = KSASolver(cfg, instance, env=env)
        solver.run()

    env.dispose()


if __name__ == "__main__":
    main()
