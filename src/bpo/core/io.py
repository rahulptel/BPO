import json
from datetime import datetime
from pathlib import Path

import torch
from botorch.utils.multi_objective.pareto import is_non_dominated


def _acquisition_directory_chain(config):
    key = config.acquisition.lower()
    if key == "random":
        return key, [
            ("n_initial_samples", config.n_initial_samples),
            ("n_iterations", config.n_iterations),
            ("batch_size_q", config.batch_size_q),
        ]
    if key == "qlogehvi":
        return key, [
            ("n_initial_samples", config.n_initial_samples),
            ("n_iterations", config.n_iterations),
            ("batch_size_q", config.batch_size_q),
            ("mc_samples", config.mc_samples),
            ("raw_samples", config.raw_samples),
        ]
    return key, [("batch_size_q", config.batch_size_q)]


def _ref_point_to_list(ref_point):
    if ref_point is None:
        return None
    if isinstance(ref_point, torch.Tensor):
        return ref_point.detach().cpu().tolist()
    return list(ref_point)


def save_result(problem, config, iteration_records, train_obj):
    final_pareto_mask = is_non_dominated(train_obj)
    final_nd_points = train_obj[final_pareto_mask].detach().cpu().tolist()

    acquisition_key, dir_chain = _acquisition_directory_chain(config)

    results = {
        "problem": problem.name,
        "problem_metadata": problem.metadata(),
        "rseed": config.rseed,
        "acquisition_function": config.acquisition,
        "acquisition_config": {
            "mc_samples": config.mc_samples,
            "batch_size_q": config.batch_size_q,
            "num_restarts": config.num_restarts,
            "raw_samples": config.raw_samples,
            "sequential": config.sequential,
        },
        "n_initial_samples": config.n_initial_samples,
        "n_iterations": config.n_iterations,
        "ref_point": _ref_point_to_list(config.ref_point),
        "iterations": iteration_records,
        "nondominated_solutions": final_nd_points,
    }

    base_dir = problem.io_base_dir(config)
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
