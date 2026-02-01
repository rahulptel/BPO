import gc
import time

import gurobipy as gp
import torch
from hydra import main as hydra_main

from problem import build_instance
from solver.ksa import KSASolver
from utils import build_gurobi_env

torch.set_default_dtype(torch.float64)


def _is_memory_error(exc):
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, gp.GurobiError):
        message = str(exc).lower()
        return "memory" in message or "memlimit" in message
    return False


def _set_process_memory_limit_gb(limit_gb):
    if limit_gb is None:
        return
    try:
        import resource
    except ImportError:
        print("resource module not available; skipping process memory limit.")
        return

    limit_bytes = int(float(limit_gb) * 1024**3)
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    new_soft = limit_bytes
    if hard == resource.RLIM_INFINITY:
        new_hard = limit_bytes
    else:
        new_soft = min(limit_bytes, hard)
        new_hard = min(hard, limit_bytes)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (new_soft, new_hard))
        print(f"Process memory limit set to ~{limit_gb} GB.")
    except (ValueError, OSError) as exc:
        print(f"Failed to set process memory limit: {exc}")


@hydra_main(config_path="configs", config_name="run_ksa", version_base=None)
def main(cfg):
    if cfg.optimizer != "gurobi":
        raise ValueError("KSA currently supports only the Gurobi optimizer.")

    _set_process_memory_limit_gb(getattr(cfg, "mem_limit_gb", 16))

    t0 = time.time()
    env = build_gurobi_env(
        time_limit=cfg.time_limit,
        output_flag=cfg.outputflag,
        mipgap=cfg.mipgap,
    )
    env.start()
    print("Time taken to start gurobi env: ", time.time() - t0)

    for pid in range(cfg.from_pid, cfg.to_pid):
        instance = None
        solver = None
        try:
            cfg.problem.iseed = pid
            instance = build_instance(cfg.problem, env, cfg.optimizer)
            solver = KSASolver(cfg, instance, env=env)
            solver.run()
        except (gp.GurobiError, MemoryError) as exc:
            if _is_memory_error(exc):
                print(
                    f"Skipping pid={pid}: memory limit reached. "
                    f"Details: {exc}"
                )
                continue
            raise
        finally:
            instance = None
            solver = None
            gc.collect()

    env.dispose()


if __name__ == "__main__":
    main()
