from dataclasses import dataclass, field

import torch
from hydra import main as hydra_main
from omegaconf import OmegaConf

from acquisition import available_acquisitions
from bpo.core.model import available_surrogates
from bpo.core.run import run_bo
from bpo.problems import available_problems

torch.set_default_dtype(torch.float64)


@dataclass
class ProblemConfig:
    __annotations__ = {
        "iseed": object,
        "rho": object,
        "ref_point": object,
    }
    iseed = 123
    rho = 1e-4
    ref_point = None


@dataclass
class KnapsackProblemConfig(ProblemConfig):
    __annotations__ = {
        "name": object,
        "n_items": object,
        "n_objs": object,
        "density": object,
    }
    name = "mokp"
    n_items = 50
    n_objs = 3
    density = 0.5


@dataclass
class SurrogateConfig:
    __annotations__ = {
        "name": object,
    }
    name = "gp"


@dataclass
class GPSurrogateConfig(SurrogateConfig):
    __annotations__ = {
        "name": object,
    }
    name = "gp"


@dataclass
class IBNNSurrogateConfig(SurrogateConfig):
    __annotations__ = {
        "name": object,
    }
    name = "ibnn"


@dataclass
class NoneSurrogateConfig(SurrogateConfig):
    __annotations__ = {
        "name": object,
    }
    name = "none"


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
        "surrogate": object,
    }
    problem = field(default_factory=KnapsackProblemConfig)
    surrogate = field(default_factory=GPSurrogateConfig)
    acquisition = field(default_factory=AcquisitionConfig)
    bo = field(default_factory=BOConfig)


_PROBLEM_CONFIG_BUILDERS = {
    "mokp": KnapsackProblemConfig,
}


_SURROGATE_CONFIG_BUILDERS = {
    "gp": GPSurrogateConfig,
    "ibnn": IBNNSurrogateConfig,
    "none": NoneSurrogateConfig,
}


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
    if config.surrogate.name not in available_surrogates():
        raise ValueError(
            f"Unknown surrogate '{config.surrogate.name}'. "
            f"Available surrogates: {', '.join(available_surrogates())}"
        )


def _build_mokp_problem(cfg):
    problem_cfg = cfg.problem
    from bpo.problems.mokp import MOKP

    return MOKP(
        n_items=problem_cfg.n_items,
        n_objs=problem_cfg.n_objs,
        density=problem_cfg.density,
        iseed=problem_cfg.iseed,
        rho=problem_cfg.rho,
    )


def _build_problem(cfg):
    builders = {
        "mokp": _build_mokp_problem,
    }

    problem_name = cfg.problem.name.lower()
    if problem_name not in builders:
        available = ", ".join(sorted(builders))
        raise ValueError(
            f"Unsupported problem '{cfg.problem.name}'. Supported problems: {available}"
        )

    return builders[problem_name](cfg)


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
        problem_cfg = cfg_dict.get("problem", {})
        if isinstance(problem_cfg, dict):
            problem_name = problem_cfg.get("name", config.problem.name)
            problem_key = str(problem_name).lower()
            if problem_key not in _PROBLEM_CONFIG_BUILDERS:
                available = ", ".join(sorted(_PROBLEM_CONFIG_BUILDERS))
                raise ValueError(
                    f"Unsupported problem config '{problem_name}'. Supported problems: {available}"
                )
            config.problem = _PROBLEM_CONFIG_BUILDERS[problem_key]()
            _apply(config.problem, problem_cfg)
        _apply(config.acquisition, cfg_dict.get("acquisition", {}))
        _apply(config.bo, cfg_dict.get("bo", {}))
        surrogate_cfg = cfg_dict.get("surrogate", {})
        if isinstance(surrogate_cfg, dict):
            surrogate_name = surrogate_cfg.get("name", config.surrogate.name)
            surrogate_key = str(surrogate_name).lower()
            if surrogate_key not in _SURROGATE_CONFIG_BUILDERS:
                available = ", ".join(sorted(_SURROGATE_CONFIG_BUILDERS))
                raise ValueError(
                    f"Unsupported surrogate config '{surrogate_name}'. Supported surrogates: {available}"
                )
            config.surrogate = _SURROGATE_CONFIG_BUILDERS[surrogate_key]()
            _apply(config.surrogate, surrogate_cfg)
    return config


@hydra_main(config_path="configs", config_name="run_bpo", version_base=None)
def main(cfg):
    config = _load_config(cfg)
    _validate_problem_config(config)

    problem = _build_problem(config)
    run_bo(problem, config)


if __name__ == "__main__":
    main()
