from dataclasses import dataclass, field

import torch
from hydra import main as hydra_main
from omegaconf import OmegaConf

from acquisition import available_acquisitions
from bpo.core.run import run_bo
from bpo.problems import available_problems, build_problem

torch.set_default_dtype(torch.float64)


@dataclass
class ProblemConfig:
    __annotations__ = {
        "name": object,
        "n_items": object,
        "n_objs": object,
        "density": object,
        "iseed": object,
        "rho": object,
        "ref_point": object,
    }
    name = "mokp"
    n_items = 50
    n_objs = 3
    density = 0.5
    iseed = 123
    rho = 1e-4
    ref_point = None


@dataclass
class AcquisitionConfig:
    __annotations__ = {
        "name": object,
        "mc_samples": object,
        "batch_size_q": object,
        "sequential": object,
    }
    name = "qlogehvi"
    mc_samples = 128
    batch_size_q = 2
    sequential = True


@dataclass
class BOConfig:
    __annotations__ = {
        "n_initial_samples": object,
        "n_iterations": object,
        "num_restarts": object,
        "raw_samples": object,
        "rseed": object,
        "should_maximize": object,
    }
    n_initial_samples = 10
    n_iterations = 20
    num_restarts = 10
    raw_samples = 512
    rseed = 123
    should_maximize = True


@dataclass
class Config:
    __annotations__ = {
        "problem": object,
        "acquisition": object,
        "bo": object,
    }
    problem = field(default_factory=ProblemConfig)
    acquisition = field(default_factory=AcquisitionConfig)
    bo = field(default_factory=BOConfig)


def _validate_problem_config(config):
    if config.problem.name not in available_problems():
        raise ValueError(
            f"Unknown problem '{config.problem.name}'. "
            f"Available problems: {', '.join(available_problems())}"
        )
    if config.acquisition.name not in available_acquisitions():
        raise ValueError(
            f"Unknown acquisition '{config.acquisition.name}'. "
            f"Available acquisitions: {', '.join(available_acquisitions())}"
        )


def _build_problem(cfg):
    return build_problem(
        cfg.problem.name,
        n_items=cfg.problem.n_items,
        n_objs=cfg.problem.n_objs,
        density=cfg.problem.density,
        iseed=cfg.problem.iseed,
        rho=cfg.problem.rho,
    )


def _load_config(cfg):
    def _apply(target, data):
        if not isinstance(data, dict):
            return
        for key, value in data.items():
            if hasattr(target, key):
                setattr(target, key, value)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    config = Config()
    if isinstance(cfg_dict, dict):
        _apply(config.problem, cfg_dict.get("problem", {}))
        _apply(config.acquisition, cfg_dict.get("acquisition", {}))
        _apply(config.bo, cfg_dict.get("bo", {}))
    return config


@hydra_main(config_path="configs", config_name="run_bpo", version_base=None)
def main(cfg):
    config = _load_config(cfg)
    _validate_problem_config(config)

    problem = _build_problem(config)
    run_bo(problem, config)


if __name__ == "__main__":
    main()
