import json
from datetime import datetime
from pathlib import Path


def _directory_chain(config):
    chain = [
        ("n_initial_samples", config.random.n_initial_samples),
        ("n_iterations", config.random.n_iterations),
        ("batch_size", config.random.batch_size_q),
    ]
    return chain


def save_result(instance, config, records, n_evaluations, ref_point, time_dict):
    result = {
        "problem": instance.name,
        "problem_metadata": instance.metadata(),
        "random_config": {
            "n_initial_samples": config.random.n_initial_samples,
            "n_iterations": config.random.n_iterations,
            "batch_size_q": config.random.batch_size_q,
            "rseed": config.random.rseed,
            "time_limit": config.random.time_limit,
            "save_prefs": config.random.save_prefs,
            "save_objs": config.random.save_objs,
        },
        "rho": config.problem.rho,
        "n_evaluations": n_evaluations,
        "ref_point": ref_point.detach().cpu().tolist(),
        "iterations": records,
        "nondominated_solutions": records[-1]["n_nd"],
        "time_dict": time_dict,
    }

    base_dir = Path("../outputs/aug_cheby")
    output_dir = base_dir / instance.base_descriptor()
    output_dir /= f"rseed-{config.random.rseed}"
    for key, value in _directory_chain(config):
        output_dir /= f"{key}-{value}"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / f"run_random_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"Saved random solver results to {output_path}")
    return output_path
