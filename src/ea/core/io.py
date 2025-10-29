import json
from datetime import datetime

import numpy as np


def _ref_point_to_list(ref_point):
    if ref_point is None:
        return None
    if isinstance(ref_point, np.ndarray):
        return ref_point.tolist()
    return list(ref_point)


def _algorithm_directory_chain(config):
    alg = config.algorithm
    # Sanitize time string for safe filesystem paths (avoid colons, etc.)
    time_str = str(alg.time)
    time_str = time_str.replace(":", "-").replace("/", "-").replace(" ", "_")
    return [
        ("algorithm", str(alg.name).lower()),
        ("pop_size", alg.pop_size),
        ("time", time_str),
    ]


def save_result(
    problem,
    config,
    y_sol_nd,
    x_sol_nd,
    hv,
    n_nd,
    ref_point,
    time_dict,
    generation_records=None,
):
    algorithm_config = {}
    if hasattr(config.algorithm, "__dict__"):
        algorithm_config = {key: value for key, value in vars(config.algorithm).items()}

    results = {
        "problem": problem.name,
        "problem_metadata": problem.metadata(),
        "rseed": config.algorithm.seed,
        "algorithm": algorithm_config,
        "x_sol_nd": x_sol_nd,
        "y_sol_nd": y_sol_nd,
        "hypervolume": hv,
        "n_nd": n_nd,
        "ref_point": _ref_point_to_list(ref_point),
        "time_dict": time_dict,
        "generations": generation_records or [],
    }

    base_dir = problem.io_base_dir(config.algorithm)
    output_dir = base_dir
    for key, value in _algorithm_directory_chain(config):
        output_dir /= f"{key}-{value}"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / f"run_ea_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Saved EA results to {output_path}")
    return output_path
