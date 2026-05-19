#!/usr/bin/env python3
"""Simulate parallel MOAP/MOKP performance by unioning seed-wise solutions."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

try:
    import pygmo as pg
except Exception:
    print("pygmo not available")

MOAP_PART_RE = re.compile(r"moap-agents-(\d+)_objs-(\d+)_iseed-(\d+)")
MOKP_PART_RE = re.compile(r"mokp-items-(\d+)_objs-(\d+)_iseed-(\d+)")
N_INIT_RE = re.compile(r"(?:^|/)n_init-(\d+)(?:/|$)")

MOAP_OBJ_AGENT_TIME = [
    (3, 175, 240),
    (3, 200, 480),
    (4, 100, 240),
    (4, 125, 480),
    (5, 75, 480),
    (6, 50, 240),
    (7, 50, 480),
]
MOKP_OBJ_ITEMS_TIME = [
    (3, 1000, 240),
    (3, 1250, 480),
    (4, 750, 240),
    (4, 1000, 480),
    (5, 500, 240),
    (5, 750, 480),
    (6, 500, 240),
]

METHOD_ORDER = {"ea": 0, "aug_cheby": 1, "bpo": 2}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Simulate k-way parallel MOAP/MOKP runs by unioning objective sets "
            "from seeds 1..k and recomputing final HV."
        )
    )
    parser.add_argument(
        "--outputs-dir",
        default=None,
        help="Path to raw outputs/ directory (default: repository outputs/).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["ea", "aug_cheby", "bpo"],
        help="Methods to include (default: ea aug_cheby bpo).",
    )
    parser.add_argument(
        "--iseed-min",
        type=int,
        default=26,
        help="Minimum instance seed (default: 26).",
    )
    parser.add_argument(
        "--iseed-max",
        type=int,
        default=50,
        help="Maximum instance seed (default: 50).",
    )
    parser.add_argument(
        "--rseed-min",
        type=int,
        default=1,
        help="Minimum run seed (default: 1).",
    )
    parser.add_argument(
        "--rseed-max",
        type=int,
        default=5,
        help="Maximum run seed (default: 5).",
    )
    parser.add_argument(
        "--n-parallel-max",
        type=int,
        default=5,
        help="Maximum simulated parallel workers k (default: 5).",
    )
    parser.add_argument(
        "--hv-lib",
        choices=["pygmo", "botorch"],
        default="pygmo",
        help="Hypervolume backend (default: pygmo).",
    )
    parser.add_argument(
        "--problem",
        choices=["mokp", "moap", "both"],
        default="both",
        help="Problem to simulate (default: both).",
    )
    parser.add_argument(
        "--mokp-save-csv",
        default=None,
        help="Path to write MOKP summary CSV (default: results/mokp_parallel_result.csv).",
    )
    parser.add_argument(
        "--mokp-save-instance-csv",
        default=None,
        help="Path to write per-instance MOKP HV CSV (default: results/mokp_parallel_instance_result.csv).",
    )
    parser.add_argument(
        "--moap-save-csv",
        default=None,
        help="Path to write MOAP summary CSV (default: results/moap_parallel_result.csv).",
    )
    parser.add_argument(
        "--moap-save-instance-csv",
        default=None,
        help="Path to write per-instance MOAP HV CSV (default: results/moap_parallel_instance_result.csv).",
    )
    return parser.parse_args()


def _parse_moap_descriptor_from_path(path):
    for part in path.parts:
        match = MOAP_PART_RE.search(part)
        if match:
            return {
                "agents": int(match.group(1)),
                "objs": int(match.group(2)),
                "iseed": int(match.group(3)),
                "instance_part": part,
            }
    return None


def _parse_mokp_descriptor_from_path(path):
    for part in path.parts:
        match = MOKP_PART_RE.search(part)
        if match:
            return {
                "items": int(match.group(1)),
                "objs": int(match.group(2)),
                "iseed": int(match.group(3)),
                "instance_part": part,
            }
    return None


def _parse_rseed_from_path(path):
    for part in path.parts:
        if part.startswith("rseed-"):
            value = part.split("rseed-")[-1]
            try:
                return int(value)
            except ValueError:
                return None
        if "_seed-" in part:
            value = part.split("_seed-")[-1]
            digits = []
            for char in value:
                if char.isdigit():
                    digits.append(char)
                else:
                    break
            if digits:
                return int("".join(digits))
    return None


def _parse_time_limit_from_path(path):
    for part in path.parts:
        if part.startswith("time-"):
            value = part.split("time-")[-1]
            if value.isdigit():
                return int(value)
            pieces = value.split("-")
            if len(pieces) == 3 and all(piece.isdigit() for piece in pieces):
                hours, minutes, seconds = (int(piece) for piece in pieces)
                return hours * 3600 + minutes * 60 + seconds
    return None


def _in_range(value, low, high):
    if value is None:
        return False
    if low is not None and value < low:
        return False
    if high is not None and value > high:
        return False
    return True


def _csv_escape(value):
    text = str(value)
    if "," in text or '"' in text or "\n" in text:
        text = text.replace('"', '""')
        return f'"{text}"'
    return text


def config_signature(method_root, run_path, instance_part):
    rel = run_path.relative_to(method_root)
    parts = list(rel.parts)
    if not parts:
        return "-"

    if parts[0] == instance_part:
        parts = parts[1:]
    if parts and parts[-1].endswith(".json"):
        parts = parts[:-1]
    parts = [part for part in parts if not part.startswith("rseed-")]

    return "/".join(parts) if parts else "-"


def find_run_files(outputs_dir, method):
    root = outputs_dir / method
    if method == "bpo":
        pattern = "run_bo_*.json"
    elif method == "aug_cheby":
        pattern = "run_aug_cheby_*.json"
    elif method == "ksa":
        pattern = "run_ksa_*.json"
    elif method == "ea":
        pattern = "run_ea_*.json"
    else:
        pattern = "run_*.json"

    if not root.exists():
        return root, []
    return root, sorted(root.rglob(pattern))


def parse_n_init(config):
    if not config:
        return None
    match = N_INIT_RE.search(str(config))
    if match:
        return match.group(1)
    return None


def method_label(method, config):
    normalized = str(method).strip().lower()
    if normalized == "ea":
        return "NSGA-II"
    if normalized == "ksa":
        return "KSA"
    if normalized == "aug_cheby":
        return "ATS"
    if normalized == "bpo":
        n_init = parse_n_init(config)
        return f"BPO-{n_init}" if n_init is not None else "BPO"
    return str(method).strip()


def normalize_hypervolume(unnorm_hv, ideal_point):
    if ideal_point is None:
        return unnorm_hv

    denom = np.abs(ideal_point).prod()
    if denom == 0:
        return unnorm_hv

    return float(unnorm_hv / denom)


def compute_hypervolume_botorch(Y_nd, ref_point, ideal_point=None, normalize=True):
    Y_nd, ref_point = (
        torch.from_numpy(Y_nd, dtype=torch.get_default_dtype()),
        torch.from_numpy(ref_point, dtype=torch.get_default_dtype()),
    )
    bd = FastNondominatedPartitioning(ref_point=ref_point, Y=Y_nd)
    hv_val = bd.compute_hypervolume().item()
    if normalize and ideal_point is not None:
        return normalize_hypervolume(hv_val, ref_point - ideal_point)
    return float(hv_val)


def compute_hypervolume_pygmo(
    points,
    ref_point,
    ideal_point=None,
    normalize=True,
    approx=False,
    eps=0.1,
    delta=0.1,
):
    hv = pg.hypervolume(points)
    hv_val = (
        hv.compute(ref_point, hv_algo=pg.bf_fpras(eps=eps, delta=delta))
        if approx
        else hv.compute(ref_point)
    )
    return (
        normalize_hypervolume(hv_val, ref_point - ideal_point) if normalize else hv_val
    )


def compute_hypervolume(
    points,
    ref_point,
    ideal_point=None,
    normalize=True,
    approx=False,
    eps=0.1,
    delta=0.1,
    lib="pygmo",
):
    if lib == "pygmo":
        return compute_hypervolume_pygmo(
            points,
            ref_point,
            ideal_point=ideal_point,
            normalize=normalize,
            approx=approx,
            eps=eps,
            delta=delta,
        )
    return compute_hypervolume_botorch(
        points, ref_point, ideal_point=ideal_point, normalize=normalize
    )


def build_allowed_specs(obj_dim_time):
    allowed_specs = defaultdict(set)
    size_order = []
    seen = set()
    for n_objs, dim_value, time_limit in obj_dim_time:
        key = (int(n_objs), int(dim_value))
        allowed_specs[key].add(int(time_limit))
        if key not in seen:
            size_order.append(key)
            seen.add(key)
    return allowed_specs, size_order


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_run_meta(problem_spec, method_root, run_path, allowed_specs, args):
    desc = problem_spec["parse_descriptor"](run_path)
    if desc is None:
        return None

    time_limit = _parse_time_limit_from_path(run_path)
    if time_limit is None:
        return None
    allowed_times = allowed_specs.get((desc["objs"], desc[problem_spec["dim_label"]]))
    if not allowed_times or time_limit not in allowed_times:
        return None

    rseed = _parse_rseed_from_path(run_path)
    if rseed is None:
        return None
    if not _in_range(desc["iseed"], args.iseed_min, args.iseed_max):
        return None
    if not _in_range(rseed, args.rseed_min, args.rseed_max):
        return None

    config = config_signature(method_root, run_path, desc["instance_part"])
    return {
        "n_objs": int(desc["objs"]),
        problem_spec["dim_key"]: int(desc[problem_spec["dim_label"]]),
        "iseed": int(desc["iseed"]),
        "rseed": int(rseed),
        "config": config,
        "instance_part": desc["instance_part"],
    }


def choose_latest(existing_path, candidate_path):
    if existing_path is None:
        return candidate_path
    if candidate_path.name > existing_path.name:
        return candidate_path
    return existing_path


def collect_runs(outputs_dir, methods, allowed_specs, args, problem_spec):
    selected_paths = {}
    for method in methods:
        method_root, run_files = find_run_files(outputs_dir, method)
        for run_path in run_files:
            meta = parse_run_meta(problem_spec, method_root, run_path, allowed_specs, args)
            if meta is None:
                continue
            key = (
                method,
                meta["n_objs"],
                meta[problem_spec["dim_key"]],
                meta["config"],
                meta["iseed"],
                meta["rseed"],
            )
            selected_paths[key] = choose_latest(selected_paths.get(key), run_path)

    runs = {}
    for key, run_path in selected_paths.items():
        payload = load_json(run_path)
        objs = payload.get("objs")
        if not isinstance(objs, list) or len(objs) == 0:
            continue
        runs[key] = np.asarray(objs, dtype=np.float64)
    return runs


def collect_instance_points(outputs_dir, allowed_specs, args, problem_spec):
    method_root, run_files = find_run_files(outputs_dir, "aug_cheby")
    point_paths = {}
    for run_path in run_files:
        meta = parse_run_meta(problem_spec, method_root, run_path, allowed_specs, args)
        if meta is None:
            continue
        key = (
            meta["n_objs"],
            meta[problem_spec["dim_key"]],
            meta["iseed"],
        )
        point_paths[key] = choose_latest(point_paths.get(key), run_path)

    points = {}
    for key, run_path in point_paths.items():
        payload = load_json(run_path)
        ref_point = payload.get("ref_point")
        ideal_point = payload.get("ideal_point")
        if not isinstance(ref_point, list) or not isinstance(ideal_point, list):
            continue
        points[key] = (
            np.asarray(ref_point, dtype=np.float64),
            np.asarray(ideal_point, dtype=np.float64),
        )
    return points


def method_sort_key(method, config):
    n_init = parse_n_init(config)
    n_init_sort = int(n_init) if n_init and n_init.isdigit() else 10**9
    return (
        METHOD_ORDER.get(method, 99),
        n_init_sort,
        str(config),
    )


def compute_union_hv(union_objs, ref_point, ideal_point, hv_lib):
    objs = np.unique(np.asarray(union_objs, dtype=np.float64), axis=0)
    nd_idx = NonDominatedSorting().do(objs, only_non_dominated_front=True)
    nd_objs = objs[nd_idx] if len(nd_idx) > 0 else objs
    hv = float(
        compute_hypervolume(
            nd_objs,
            ref_point,
            ideal_point=ideal_point,
            normalize=True,
            approx=False,
            eps=0.1,
            delta=0.1,
            lib=hv_lib,
        )
    )
    n_nd = len(nd_idx) if len(nd_idx) > 0 else len(objs)
    return hv, int(n_nd)


def summarize_values(values):
    if not values:
        return None, None, None, None
    mu = float(np.mean(values))
    if len(values) < 2:
        sigma = 0.0
    else:
        mu2 = float(np.mean([(x - mu) ** 2 for x in values]))
        sigma = mu2**0.5
    return mu, sigma, min(values), max(values)


def write_csv(path, rows, problem_spec):
    header = [
        "problem",
        "n_objs",
        problem_spec["dim_key"],
        "size",
        "method",
        "method_raw",
        "config",
        "n_parallel",
        "hv",
        "n_instances",
        "std_hv",
        "min_hv",
        "max_hv",
        "mean_n_nd",
    ]
    lines = [",".join(header)]
    for row in rows:
        hv_value = row.get("hv")
        std_hv = row.get("std_hv")
        min_hv = row.get("min_hv")
        max_hv = row.get("max_hv")
        mean_n_nd = row.get("mean_n_nd")
        lines.append(
            ",".join(
                [
                    str(row.get("problem", "")),
                    str(row.get("n_objs", "")),
                    str(row.get(problem_spec["dim_key"], "")),
                    str(row.get("size", "")),
                    _csv_escape(row.get("method", "")),
                    str(row.get("method_raw", "")),
                    _csv_escape(row.get("config", "")),
                    str(row.get("n_parallel", "")),
                    "" if hv_value is None else f"{hv_value:.10f}",
                    str(row.get("n_instances", 0)),
                    "" if std_hv is None else f"{std_hv:.10f}",
                    "" if min_hv is None else f"{min_hv:.10f}",
                    "" if max_hv is None else f"{max_hv:.10f}",
                    "" if mean_n_nd is None else f"{mean_n_nd:.6f}",
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_instance_csv(path, rows, problem_spec):
    header = [
        "problem",
        "n_objs",
        problem_spec["dim_key"],
        "size",
        "method",
        "method_raw",
        "config",
        "n_parallel",
        "iseed",
        "hv",
        "n_points_union",
    ]
    lines = [",".join(header)]
    for row in rows:
        lines.append(
            ",".join(
                [
                    str(row.get("problem", "")),
                    str(row.get("n_objs", "")),
                    str(row.get(problem_spec["dim_key"], "")),
                    str(row.get("size", "")),
                    _csv_escape(row.get("method", "")),
                    str(row.get("method_raw", "")),
                    _csv_escape(row.get("config", "")),
                    str(row.get("n_parallel", "")),
                    str(row.get("iseed", "")),
                    f"{row.get('hv', 0.0):.10f}",
                    str(row.get("n_points_union", "")),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def simulate_parallel(problem_spec, args, outputs_dir, save_csv, save_instance_csv):
    allowed_specs, size_order = build_allowed_specs(problem_spec["obj_dim_time"])
    runs = collect_runs(outputs_dir, args.methods, allowed_specs, args, problem_spec)
    points = collect_instance_points(outputs_dir, allowed_specs, args, problem_spec)

    if not runs:
        print(f"[{problem_spec['problem']}] No matching runs found.")
        return

    max_parallel = min(
        int(args.n_parallel_max),
        int(args.rseed_max) - int(args.rseed_min) + 1,
    )
    group_keys = defaultdict(set)
    for method, n_objs, dim_value, config, _, _ in runs.keys():
        group_keys[(n_objs, dim_value)].add((method, config))

    summary_rows = []
    instance_rows = []
    size_rank = {key: idx for idx, key in enumerate(size_order)}

    for n_objs, dim_value in size_order:
        methods_here = sorted(
            group_keys.get((n_objs, dim_value), set()),
            key=lambda item: method_sort_key(item[0], item[1]),
        )
        if not methods_here:
            continue

        for method, config in methods_here:
            label = method_label(method, config)
            for k in range(1, max_parallel + 1):
                hvs = []
                nd_counts = []
                for iseed in range(int(args.iseed_min), int(args.iseed_max) + 1):
                    point_key = (n_objs, dim_value, iseed)
                    if point_key not in points:
                        continue
                    ref_point, ideal_point = points[point_key]

                    seed_values = list(range(int(args.rseed_min), int(args.rseed_min) + k))
                    objs_union = []
                    for rseed in seed_values:
                        run_key = (method, n_objs, dim_value, config, iseed, rseed)
                        objs = runs.get(run_key)
                        if objs is None:
                            objs_union = []
                            break
                        objs_union.append(objs)
                    if not objs_union:
                        continue

                    union_array = np.concatenate(objs_union, axis=0)
                    hv, n_nd = compute_union_hv(
                        union_array, ref_point, ideal_point, args.hv_lib
                    )
                    hvs.append(hv)
                    nd_counts.append(n_nd)
                    instance_rows.append(
                        {
                            "problem": problem_spec["problem"],
                            "n_objs": n_objs,
                            problem_spec["dim_key"]: dim_value,
                            "size": problem_spec["size_fmt"].format(
                                n_objs=n_objs, dim_value=dim_value
                            ),
                            "method": label,
                            "method_raw": method,
                            "config": config,
                            "n_parallel": k,
                            "iseed": iseed,
                            "hv": hv,
                            "n_points_union": int(len(union_array)),
                        }
                    )

                mean_hv, std_hv, min_hv, max_hv = summarize_values(hvs)
                mean_n_nd = float(np.mean(nd_counts)) if nd_counts else None
                summary_rows.append(
                    {
                        "problem": problem_spec["problem"],
                        "n_objs": n_objs,
                        problem_spec["dim_key"]: dim_value,
                        "size": problem_spec["size_fmt"].format(
                            n_objs=n_objs, dim_value=dim_value
                        ),
                        "method": label,
                        "method_raw": method,
                        "config": config,
                        "n_parallel": k,
                        "hv": mean_hv,
                        "std_hv": std_hv,
                        "min_hv": min_hv,
                        "max_hv": max_hv,
                        "mean_n_nd": mean_n_nd,
                        "n_instances": len(hvs),
                    }
                )

    summary_rows = sorted(
        summary_rows,
        key=lambda row: (
            size_rank.get((row["n_objs"], row[problem_spec["dim_key"]]), 999),
            method_sort_key(row["method_raw"], row["config"]),
            row["n_parallel"],
        ),
    )

    if not summary_rows:
        print(f"[{problem_spec['problem']}] No simulated rows were produced.")
        return

    print(f"\n=== {problem_spec['problem'].upper()} ===")
    for row in summary_rows:
        hv_text = "--" if row["hv"] is None else f"{row['hv']:.6f}"
        print(
            f"{row['size']:>20} | {row['method']:<10} | "
            f"k={row['n_parallel']} | hv={hv_text} | n={row['n_instances']}"
        )

    save_csv = Path(save_csv).expanduser().resolve()
    write_csv(save_csv, summary_rows, problem_spec)
    print(f"\nWrote CSV: {save_csv}")

    if save_instance_csv:
        save_instance_csv = Path(save_instance_csv).expanduser().resolve()
        write_instance_csv(save_instance_csv, instance_rows, problem_spec)
        print(f"Wrote per-instance CSV: {save_instance_csv}")


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = (
        Path(args.outputs_dir).expanduser().resolve()
        if args.outputs_dir
        else repo_root / "outputs"
    )

    mokp_spec = {
        "problem": "mokp",
        "dim_key": "n_items",
        "dim_label": "items",
        "obj_dim_time": MOKP_OBJ_ITEMS_TIME,
        "parse_descriptor": _parse_mokp_descriptor_from_path,
        "size_fmt": "objs-{n_objs}_items-{dim_value}",
    }
    moap_spec = {
        "problem": "moap",
        "dim_key": "n_agents",
        "dim_label": "agents",
        "obj_dim_time": MOAP_OBJ_AGENT_TIME,
        "parse_descriptor": _parse_moap_descriptor_from_path,
        "size_fmt": "objs-{n_objs}_agents-{dim_value}",
    }

    if args.problem in ("mokp", "both"):
        simulate_parallel(
            mokp_spec,
            args,
            outputs_dir,
            args.mokp_save_csv or results_dir / "mokp_parallel_result.csv",
            args.mokp_save_instance_csv
            or results_dir / "mokp_parallel_instance_result.csv",
        )

    if args.problem in ("moap", "both"):
        simulate_parallel(
            moap_spec,
            args,
            outputs_dir,
            args.moap_save_csv or results_dir / "moap_parallel_result.csv",
            args.moap_save_instance_csv
            or results_dir / "moap_parallel_instance_result.csv",
        )


if __name__ == "__main__":
    main()
