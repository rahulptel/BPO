from dataclasses import dataclass, field

from hydra import main as hydra_main
from omegaconf import OmegaConf

from ga.core.run import run_ga as execute_ga
from ga.problems import available_problems


@dataclass
class ProblemConfig:
    __annotations__ = {
        "iseed": object,
        "ref_point": object,
    }
    iseed = 123
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
class AlgorithmConfig:
    __annotations__ = {
        "pop_size": object,
        "seed": object,
        "eliminate_duplicates": object,
        "time": object,
    }
    pop_size = 200
    seed = 123
    eliminate_duplicates = True
    time = "00:05:00"


@dataclass
class NSGA2AlgorithmConfig(AlgorithmConfig):
    __annotations__ = {"name": object}
    name = "nsga2"


@dataclass
class NSGA3AlgorithmConfig(AlgorithmConfig):
    __annotations__ = {"name": object, "ref_dir_method": object, "n_partitions": object}
    name = "nsga3"
    ref_dir_method = "dan-dennis"
    n_partitions = 10


@dataclass
class SMSEMOAAlgorithmConfig(AlgorithmConfig):
    __annotations__ = {"name": object}
    name = "smsemoa"


@dataclass
class CTAEAAlgorithmConfig(AlgorithmConfig):
    __annotations__ = {"name": object, "ref_dir_method": object, "n_partitions": object}
    name = "ctaea"
    ref_dir_method = "dan-dennis"
    n_partitions = 10


@dataclass
class Config:
    __annotations__ = {
        "problem": object,
        "algorithm": object,
    }
    problem = field(default_factory=KnapsackProblemConfig)
    algorithm = field(default_factory=AlgorithmConfig)


_PROBLEM_CONFIG_BUILDERS = {
    "mokp": KnapsackProblemConfig,
}


_ALGORITHM_CONFIG_BUILDERS = {
    "nsga2": NSGA2AlgorithmConfig,
    "nsga3": NSGA3AlgorithmConfig,
    "smsemoa": SMSEMOAAlgorithmConfig,
    "ctaea": CTAEAAlgorithmConfig,
}


def validate_problem_config(config):
    if config.problem.name not in available_problems():
        raise ValueError(
            f"Unknown problem '{config.problem.name}'. "
            f"Available problems: {', '.join(available_problems())}"
        )
    algorithm_name = config.algorithm.name.lower()
    if algorithm_name not in _ALGORITHM_CONFIG_BUILDERS:
        available = ", ".join(sorted(_ALGORITHM_CONFIG_BUILDERS))
        raise ValueError(
            f"Unknown algorithm '{config.algorithm.name}'. Supported algorithms: {available}"
        )


def _build_mokp_problem(cfg):
    problem_cfg = cfg.problem
    from ga.problems.mokp import MOKP

    return MOKP(
        n_items=problem_cfg.n_items,
        n_objs=problem_cfg.n_objs,
        density=problem_cfg.density,
        iseed=problem_cfg.iseed,
    )


def build_problem(cfg):
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


def load_config(cfg):
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
            problem_name = problem_cfg.get("name") or config.problem.name
            problem_key = str(problem_name).lower()
            if problem_key not in _PROBLEM_CONFIG_BUILDERS:
                available = ", ".join(sorted(_PROBLEM_CONFIG_BUILDERS))
                raise ValueError(
                    f"Unsupported problem config '{problem_name}'. Supported problems: {available}"
                )
            config.problem = _PROBLEM_CONFIG_BUILDERS[problem_key]()
            _apply(config.problem, problem_cfg)
        algorithm_cfg = cfg_dict.get("algorithm", {})
        if isinstance(algorithm_cfg, dict):
            # Avoid accessing config.algorithm.name since base AlgorithmConfig has no 'name'
            algorithm_name = algorithm_cfg.get("name") or getattr(
                config.algorithm, "name", None
            )
            algorithm_key = str(algorithm_name).lower()
            if algorithm_key not in _ALGORITHM_CONFIG_BUILDERS:
                available = ", ".join(sorted(_ALGORITHM_CONFIG_BUILDERS))
                raise ValueError(
                    f"Unsupported algorithm config '{algorithm_name}'. Supported algorithms: {available}"
                )
            config.algorithm = _ALGORITHM_CONFIG_BUILDERS[algorithm_key]()
            _apply(config.algorithm, algorithm_cfg)
        elif isinstance(algorithm_cfg, str):
            algorithm_key = algorithm_cfg.lower()
            if algorithm_key not in _ALGORITHM_CONFIG_BUILDERS:
                available = ", ".join(sorted(_ALGORITHM_CONFIG_BUILDERS))
                raise ValueError(
                    f"Unsupported algorithm config '{algorithm_cfg}'. Supported algorithms: {available}"
                )
            config.algorithm = _ALGORITHM_CONFIG_BUILDERS[algorithm_key]()
    return config


@hydra_main(config_path="configs", config_name="run_ga", version_base=None)
def main(cfg):
    config = load_config(cfg)
    validate_problem_config(config)

    problem = build_problem(config)
    execute_ga(problem, config)


if __name__ == "__main__":
    main()
