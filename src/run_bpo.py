import time

import os
import torch
from hydra import main as hydra_main

from problem import build_instance
from scalarization import build_scalarizer
from solver.bpo import BPOSolver
from utils import build_gurobi_env


def _resolve_runtime_dtype(runtime_cfg):
    if runtime_cfg is None:
        return torch.float64

    dtype_name = str(getattr(runtime_cfg, "bo_dtype", "float64")).lower()
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float64":
        return torch.float64
    raise ValueError(
        f"Unsupported runtime.bo_dtype='{dtype_name}'. Use float32 or float64."
    )


def _resolve_torch_num_threads(runtime_cfg):
    if runtime_cfg is None:
        return 1
    value = getattr(runtime_cfg, "torch_num_threads", 1)
    return max(1, int(value))


def _resolve_gurobi_threads(runtime_cfg):
    if runtime_cfg is None:
        return 1
    value = getattr(runtime_cfg, "gurobi_threads", 1)
    return max(1, int(value))


@hydra_main(config_path="configs", config_name="run_bpo", version_base=None)
def main(cfg):
    runtime_cfg = getattr(cfg, "runtime", None)
    torch_dtype = _resolve_runtime_dtype(runtime_cfg)
    torch_num_threads = _resolve_torch_num_threads(runtime_cfg)
    gurobi_threads = _resolve_gurobi_threads(runtime_cfg)

    os.environ["OMP_NUM_THREADS"] = str(torch_num_threads)
    torch.set_num_threads(torch_num_threads)
    torch.set_default_dtype(torch_dtype)

    t0 = time.time()

    env = (
        build_gurobi_env(
            time_limit=cfg.time_limit,
            output_flag=cfg.outputflag,
            mipgap=cfg.mipgap,
            threads=gurobi_threads,
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
