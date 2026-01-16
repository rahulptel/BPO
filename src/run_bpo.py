import os

os.environ["OMP_NUM_THREADS"] = "1"


import time

import torch

torch.set_num_threads(1)
from hydra import main as hydra_main

from problem import build_instance
from scalarization import build_scalarizer
from solver.bpo import BPOSolver
from utils import build_gurobi_env

torch.set_default_dtype(torch.float64)


@hydra_main(config_path="configs", config_name="run_bpo", version_base=None)
def main(cfg):
    t0 = time.time()

    env = (
        build_gurobi_env(
            time_limit=cfg.time_limit, output_flag=cfg.outputflag, mipgap=cfg.mipgap
        )
        if cfg.optimizer == "gurobi"
        else None
    )
    if env is not None:
        env.start()
    print("Time taken to start gurobi env: ", time.time() - t0)

    for pid in range(cfg.from_pid, cfg.to_pid):
        cfg.problem.iseed = pid
        instance = build_instance(cfg.problem, env, cfg.optimizer)
        scalarizer = build_scalarizer(cfg, instance, env=env)
        solver = BPOSolver(cfg, instance, scalarizer)
        solver.run()

    if env is not None:
        env.dispose()


if __name__ == "__main__":
    main()
