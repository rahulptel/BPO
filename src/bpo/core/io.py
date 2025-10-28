import json
from datetime import datetime
from pathlib import Path

import torch
from botorch.utils.multi_objective.pareto import is_non_dominated


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


def _ref_point_to_list(ref_point):
    if ref_point is None:
        return None
    if isinstance(ref_point, torch.Tensor):
        return ref_point.detach().cpu().tolist()
    return list(ref_point)


def save_result(
    problem,
    config,
    iteration_records,
    train_obj,
    ref_point,
    time_data_collection,
    time_iterations,
):
    final_pareto_mask = is_non_dominated(train_obj)
    final_nd_points = train_obj[final_pareto_mask].detach().cpu().tolist()

    acquisition_key, dir_chain = _acquisition_directory_chain(config)
    dir_chain.append(("surrogate", config.surrogate.name))

    surrogate_config = {}
    if hasattr(config.surrogate, "__dict__"):
        surrogate_config = {
            key: value
            for key, value in vars(config.surrogate).items()
            if key != "name" and not key.startswith("_")
        }

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
        "ref_point": _ref_point_to_list(ref_point),
        "iterations": iteration_records,
        "nondominated_solutions": final_nd_points,
        "time_data_collection": float(time_data_collection),
        "time_iterations": float(time_iterations),
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
