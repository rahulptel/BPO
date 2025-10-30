import json
from datetime import datetime
from pathlib import Path

import torch
from botorch.utils.multi_objective.pareto import is_non_dominated
from omegaconf import DictConfig, OmegaConf


def _acquisition_directory_chain(config):
    key = config.acquisition.name.lower()
    if key == "random":
        return key, [
            ("n_initial_samples", config.bo.n_initial_samples),
            ("n_iterations", config.bo.n_iterations),
            ("batch_size_q", config.acquisition.batch_size_q),
        ]
    if key == "qlogehvi":
        return key, [
            ("n_initial_samples", config.bo.n_initial_samples),
            ("n_iterations", config.bo.n_iterations),
            ("batch_size_q", config.acquisition.batch_size_q),
            ("mc_samples", config.acquisition.mc_samples),
            ("raw_samples", config.bo.raw_samples),
        ]
    return key, [("batch_size_q", config.acquisition.batch_size_q)]


def _surrogate_config_to_dict(surrogate_cfg):
    if surrogate_cfg is None:
        return {}
    if isinstance(surrogate_cfg, DictConfig):
        data = OmegaConf.to_container(surrogate_cfg, resolve=True)
    elif isinstance(surrogate_cfg, dict):
        data = dict(surrogate_cfg)
    elif hasattr(surrogate_cfg, "__dict__"):
        data = {
            key: value
            for key, value in vars(surrogate_cfg).items()
            if not key.startswith("_")
        }
    else:
        return {}
    if isinstance(data, dict):
        data.pop("name", None)
    return data


def save_result(
    problem,
    config,
    records,
    ref_point,
    time_dict,
):
    acquisition_key, dir_chain = _acquisition_directory_chain(config)
    dir_chain.append(("surrogate", config.surrogate.name))

    surrogate_config = _surrogate_config_to_dict(config.surrogate)

    results = {
        "problem": problem.name,
        "problem_metadata": problem.metadata(),
        "rseed": config.bo.rseed,
        "acquisition_function": config.acquisition.name,
        "acquisition_config": {
            "mc_samples": config.acquisition.mc_samples,
            "batch_size_q": config.acquisition.batch_size_q,
            "num_restarts": config.bo.num_restarts,
            "raw_samples": config.bo.raw_samples,
            "sequential": config.acquisition.sequential,
        },
        "surrogate": {
            "name": config.surrogate.name,
            "config": surrogate_config,
        },
        "n_initial_samples": config.bo.n_initial_samples,
        "n_iterations": config.bo.n_iterations,
        "ref_point": ref_point.detach().cpu().tolist(),
        "iterations": records,
        "nondominated_solutions": records[-1]["n_nd"],
        "time_dict": time_dict,
    }

    base_dir = problem.io_base_dir(config.bo)
    acquisition_dir = base_dir / acquisition_key
    output_dir = acquisition_dir
    for name, value in dir_chain:
        output_dir /= f"{name}-{value}"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / f"run_bo_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Saved BO results to {output_path}")
    return output_path
