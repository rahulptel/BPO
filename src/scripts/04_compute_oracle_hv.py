#!/usr/bin/env python3
"""Compute oracle hypervolume across all methods for MOAP and MOKP."""

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def load_script_module(name, filename):
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


moap_result = load_script_module("moap_result", "01_summarize_moap_runs.py")
mokp_result = load_script_module("mokp_result", "01_summarize_mokp_runs.py")

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import compute_hypervolume


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute oracle HV by unioning final objective sets across all methods."
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
        default=["ea", "ksa", "aug_cheby", "bpo"],
        help="Methods to include (default: ea ksa aug_cheby bpo).",
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
        "--hv-lib",
        choices=["pygmo", "botorch"],
        default="pygmo",
        help="Hypervolume backend passed to utils.compute_hypervolume.",
    )
    parser.add_argument(
        "--bpo-n-init",
        nargs="+",
        default=None,
        help="Optional BPO n_init values to include (default: include all).",
    )
    parser.add_argument(
        "--save-moap-csv",
        default=None,
        help="Path to write MOAP oracle summary CSV (default: results/moap_oracle_result.csv).",
    )
    parser.add_argument(
        "--save-mokp-csv",
        default=None,
        help="Path to write MOKP oracle summary CSV (default: results/mokp_oracle_result.csv).",
    )
    parser.add_argument(
        "--save-moap-instance-csv",
        default=None,
        help="Path to write per-instance MOAP oracle CSV (default: results/moap_oracle_instance_result.csv).",
    )
    parser.add_argument(
        "--save-mokp-instance-csv",
        default=None,
        help="Path to write per-instance MOKP oracle CSV (default: results/mokp_oracle_instance_result.csv).",
    )
    return parser.parse_args()


def build_allowed_specs(spec_triplets):
    allowed_specs = defaultdict(set)
    size_order = []
    seen = set()
    for n_objs, size, time_limit in spec_triplets:
        key = (int(n_objs), int(size))
        allowed_specs[key].add(int(time_limit))
        if key not in seen:
            size_order.append(key)
            seen.add(key)
    return allowed_specs, size_order


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def choose_latest(existing_path, candidate_path):
    if existing_path is None:
        return candidate_path
    if candidate_path.name > existing_path.name:
        return candidate_path
    return existing_path


def matches_bpo_n_init(config, bpo_n_init):
    if not bpo_n_init:
        return True
    for value in bpo_n_init:
        if f"n_init-{value}" in str(config):
            return True
    return False


def compute_union_hv(union_objs, ref_point, ideal_point, hv_lib):
    objs = np.unique(np.asarray(union_objs, dtype=np.float64), axis=0)
    nd_idx = NonDominatedSorting().do(objs, only_non_dominated_front=True)
    nd_objs = objs[nd_idx] if len(nd_idx) > 0 else objs
    hv = compute_hypervolume(
        nd_objs,
        ref_point,
        ideal_point=ideal_point,
        normalize=True,
        approx=False,
        eps=0.1,
        delta=0.1,
        lib=hv_lib,
    )
    return float(hv), int(len(nd_objs))


def summarize_group(values):
    if not values:
        return None
    mu = mean(values)
    if len(values) < 2:
        sigma = 0.0
    else:
        mu2 = mean([(x - mu) ** 2 for x in values])
        sigma = mu2**0.5
    return mu, sigma, min(values), max(values)


def format_hv_percent(value):
    if value is None:
        return ""
    return f"{value}"


def format_optional_number(value, digits=6):
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def log_progress(prefix, current, total, every):
    if total == 0:
        return
    if current == total or current % every == 0:
        print(f"{prefix} {current}/{total}")


def parse_moap_meta(method_root, run_path, allowed_specs, args):
    desc = moap_result._parse_moap_descriptor_from_path(run_path)
    if desc is None:
        return None

    time_limit = moap_result._parse_time_limit_from_path(run_path)
    if time_limit is None:
        return None
    allowed_times = allowed_specs.get((desc["objs"], desc["agents"]))
    if not allowed_times or time_limit not in allowed_times:
        return None

    if not moap_result._in_range(desc["iseed"], args.iseed_min, args.iseed_max):
        return None

    rseed = moap_result._parse_rseed_from_path(run_path)
    config = moap_result.config_signature(method_root, run_path, desc["instance_part"])
    return {
        "n_objs": int(desc["objs"]),
        "size": int(desc["agents"]),
        "iseed": int(desc["iseed"]),
        "rseed": rseed,
        "config": config,
        "instance_part": desc["instance_part"],
    }


def parse_mokp_meta(method_root, run_path, allowed_specs, args):
    desc = mokp_result._parse_mokp_descriptor_from_path(run_path)
    if desc is None:
        return None

    time_limit = mokp_result._parse_time_limit_from_path(run_path)
    if time_limit is None:
        return None
    allowed_times = allowed_specs.get((desc["objs"], desc["items"]))
    if not allowed_times or time_limit not in allowed_times:
        return None

    if not mokp_result._in_range(desc["iseed"], args.iseed_min, args.iseed_max):
        return None

    rseed = mokp_result._parse_rseed_from_path(run_path)
    config = mokp_result.config_signature(method_root, run_path, desc["instance_part"])
    return {
        "n_objs": int(desc["objs"]),
        "size": int(desc["items"]),
        "iseed": int(desc["iseed"]),
        "rseed": rseed,
        "config": config,
        "instance_part": desc["instance_part"],
    }


def rseed_in_range(rseed, args, in_range_fn):
    if rseed is None:
        return True
    return in_range_fn(rseed, args.rseed_min, args.rseed_max)


def collect_union_objs(
    outputs_dir,
    methods,
    allowed_specs,
    args,
    find_run_files,
    parse_meta,
    in_range_fn,
    progress_label,
    progress_every=1000,
):
    selected_paths = {}
    for method in methods:
        method_root, run_files = find_run_files(outputs_dir, method)
        total_files = len(run_files)
        if total_files == 0:
            print(f"{progress_label}: {method} has no runs.")
            continue
        print(f"{progress_label}: scanning {method} ({total_files} files)")
        accepted = 0
        for idx, run_path in enumerate(run_files, start=1):
            meta = parse_meta(method_root, run_path, allowed_specs, args)
            if meta is None:
                log_progress(
                    f"{progress_label}: {method}",
                    idx,
                    total_files,
                    progress_every,
                )
                continue
            if method != "ksa" and not rseed_in_range(meta["rseed"], args, in_range_fn):
                log_progress(
                    f"{progress_label}: {method}",
                    idx,
                    total_files,
                    progress_every,
                )
                continue
            if method == "bpo" and not matches_bpo_n_init(
                meta["config"], args.bpo_n_init
            ):
                log_progress(
                    f"{progress_label}: {method}",
                    idx,
                    total_files,
                    progress_every,
                )
                continue
            key = (
                method,
                meta["n_objs"],
                meta["size"],
                meta["config"],
                meta["iseed"],
                meta["rseed"],
            )
            selected_paths[key] = choose_latest(selected_paths.get(key), run_path)
            accepted += 1
            log_progress(
                f"{progress_label}: {method}",
                idx,
                total_files,
                progress_every,
            )
        print(f"{progress_label}: {method} accepted {accepted}/{total_files} runs")

    union_map = defaultdict(list)
    for key, run_path in selected_paths.items():
        payload = load_json(run_path)
        objs = payload.get("objs")
        if not isinstance(objs, list) or len(objs) == 0:
            continue
        _, n_objs, size, _, iseed, _ = key
        union_map[(n_objs, size, iseed)].append(np.asarray(objs, dtype=np.float64))
    return union_map


def collect_instance_points(
    outputs_dir,
    methods,
    allowed_specs,
    args,
    find_run_files,
    parse_meta,
    in_range_fn,
    progress_label,
    progress_every=1000,
):
    preferred_methods = ["aug_cheby"] + [m for m in methods if m != "aug_cheby"]
    points = {}
    for method in preferred_methods:
        method_root, run_files = find_run_files(outputs_dir, method)
        total_files = len(run_files)
        if total_files == 0:
            print(f"{progress_label}: {method} has no runs for ref/ideal.")
            continue
        print(
            f"{progress_label}: scanning {method} for ref/ideal ({total_files} files)"
        )
        candidate_paths = {}
        processed = 0
        for run_path in run_files:
            processed += 1
            meta = parse_meta(method_root, run_path, allowed_specs, args)
            if meta is None:
                log_progress(
                    f"{progress_label}: {method}",
                    processed,
                    total_files,
                    progress_every,
                )
                continue
            if method != "ksa" and not rseed_in_range(meta["rseed"], args, in_range_fn):
                log_progress(
                    f"{progress_label}: {method}",
                    processed,
                    total_files,
                    progress_every,
                )
                continue
            if method == "bpo" and not matches_bpo_n_init(
                meta["config"], args.bpo_n_init
            ):
                log_progress(
                    f"{progress_label}: {method}",
                    processed,
                    total_files,
                    progress_every,
                )
                continue
            key = (meta["n_objs"], meta["size"], meta["iseed"])
            if key in points:
                log_progress(
                    f"{progress_label}: {method}",
                    processed,
                    total_files,
                    progress_every,
                )
                continue
            candidate_paths[key] = choose_latest(candidate_paths.get(key), run_path)
            log_progress(
                f"{progress_label}: {method}",
                processed,
                total_files,
                progress_every,
            )

        for key, run_path in candidate_paths.items():
            if key in points:
                continue
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


def write_instance_csv(path, rows, size_label):
    header = [
        "problem",
        "n_objs",
        size_label,
        "size",
        "iseed",
        "hv",
        "n_points_union",
        "n_points_nd",
    ]
    lines = [",".join(header)]
    for row in rows:
        lines.append(
            ",".join(
                [
                    str(row.get("problem", "")),
                    str(row.get("n_objs", "")),
                    str(row.get(size_label, "")),
                    str(row.get("size", "")),
                    str(row.get("iseed", "")),
                    f"{row.get('hv', 0.0):.10f}",
                    str(row.get("n_points_union", "")),
                    str(row.get("n_points_nd", "")),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_oracle_summary_moap(path, rows):
    header = [
        "n_objs",
        "n_agents",
        "method",
        "config",
        "runs",
        "unique_iseeds",
        "unique_rseeds",
        "mean_n_evals",
        "min_n_evals",
        "max_n_evals",
        "mean_n_nd",
        "mean_hv",
        "std_hv",
        "min_hv",
        "max_hv",
    ]
    lines = [",".join(header)]
    for row in rows:
        lines.append(
            ",".join(
                [
                    str(row.get("n_objs", "")),
                    str(row.get("n_agents", "")),
                    str(row.get("method", "")),
                    moap_result._csv_escape(row.get("config", "")),
                    str(row.get("runs", "")),
                    str(row.get("unique_iseeds", "")),
                    str(row.get("unique_rseeds", "")),
                    format_optional_number(row.get("mean_n_evals")),
                    format_optional_number(row.get("min_n_evals")),
                    format_optional_number(row.get("max_n_evals")),
                    format_optional_number(row.get("mean_n_nd")),
                    format_hv_percent(row.get("mean_hv")),
                    format_hv_percent(row.get("std_hv")),
                    format_hv_percent(row.get("min_hv")),
                    format_hv_percent(row.get("max_hv")),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_oracle_summary_mokp(path, rows):
    header = [
        "n_objs",
        "n_items",
        "method",
        "config",
        "runs",
        "unique_iseeds",
        "unique_rseeds",
        "mean_n_evals",
        "mean_n_nd",
        "mean_hv",
        "std_hv",
        "min_hv",
        "max_hv",
    ]
    lines = [",".join(header)]
    for row in rows:
        lines.append(
            ",".join(
                [
                    str(row.get("n_objs", "")),
                    str(row.get("n_items", "")),
                    str(row.get("method", "")),
                    mokp_result._csv_escape(row.get("config", "")),
                    str(row.get("runs", "")),
                    str(row.get("unique_iseeds", "")),
                    str(row.get("unique_rseeds", "")),
                    format_optional_number(row.get("mean_n_evals")),
                    format_optional_number(row.get("mean_n_nd")),
                    format_hv_percent(row.get("mean_hv")),
                    format_hv_percent(row.get("std_hv")),
                    format_hv_percent(row.get("min_hv")),
                    format_hv_percent(row.get("max_hv")),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_oracle_rows(problem, size_order, union_map, points, size_field):
    instance_rows = []
    summary_rows = []
    per_size = defaultdict(list)
    missing_points = 0
    empty_union = 0
    total_instances = len(union_map)
    if total_instances:
        print(
            f"{problem['name'].upper()}: computing HV for {total_instances} instances"
        )

    processed = 0
    for (n_objs, size, iseed), obj_lists in union_map.items():
        processed += 1
        point_key = (n_objs, size, iseed)
        if point_key not in points:
            missing_points += 1
            log_progress(
                f"{problem['name'].upper()}: HV",
                processed,
                total_instances,
                100,
            )
            continue
        if not obj_lists:
            empty_union += 1
            log_progress(
                f"{problem['name'].upper()}: HV",
                processed,
                total_instances,
                100,
            )
            continue

        union_array = np.concatenate(obj_lists, axis=0)
        hv, n_points_nd = compute_union_hv(
            union_array,
            points[point_key][0],
            points[point_key][1],
            problem["hv_lib"],
        )
        row = {
            "problem": problem["name"],
            "n_objs": n_objs,
            size_field: size,
            "size": problem["size_label"](n_objs, size),
            "iseed": iseed,
            "hv": hv,
            "n_points_union": int(len(union_array)),
            "n_points_nd": int(n_points_nd),
        }
        instance_rows.append(row)
        per_size[(n_objs, size)].append(
            {
                "hv": hv,
                "n_points_nd": int(n_points_nd),
                "iseed": iseed,
            }
        )
        log_progress(
            f"{problem['name'].upper()}: HV",
            processed,
            total_instances,
            100,
        )

    for n_objs, size in size_order:
        entries = per_size.get((n_objs, size), [])
        if not entries:
            continue
        hvs = [entry["hv"] for entry in entries]
        stats = summarize_group(hvs)
        if stats is None:
            continue
        mu, sigma, lo, hi = stats
        n_nds = [entry["n_points_nd"] for entry in entries]
        mean_n_nd = mean(n_nds) if n_nds else None
        summary_rows.append(
            {
                "n_objs": n_objs,
                size_field: size,
                "method": "oracle",
                "config": "-",
                "runs": len(entries),
                "unique_iseeds": len({entry["iseed"] for entry in entries}),
                "unique_rseeds": "",
                "mean_n_evals": None,
                "min_n_evals": None,
                "max_n_evals": None,
                "mean_n_nd": mean_n_nd,
                "mean_hv": mu,
                "std_hv": sigma,
                "min_hv": lo,
                "max_hv": hi,
            }
        )

    counts = {
        "instances_with_union": len(union_map),
        "instances_used": len(instance_rows),
        "missing_points": missing_points,
        "empty_union": empty_union,
    }
    return summary_rows, instance_rows, counts


def report_problem(problem_config, counts, summary_rows):
    name = str(problem_config["name"]).upper()
    print(
        f"{name}: instances_with_union={counts['instances_with_union']} | "
        f"used={counts['instances_used']} | missing_ref_ideal={counts['missing_points']} | "
        f"empty_union={counts['empty_union']}"
    )
    size_label_fn = problem_config["size_label"]
    size_field = problem_config["size_field"]
    for row in summary_rows:
        size_text = size_label_fn(row["n_objs"], row[size_field])
        print(f"  {size_text:>20} | mean_hv={row['mean_hv']:.6f} | n={row['runs']}")


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

    moap_allowed, moap_sizes = build_allowed_specs(moap_result.MOAP_OBJ_AGENT_TIME)
    mokp_allowed, mokp_sizes = build_allowed_specs(mokp_result.MOKP_OBJ_ITEMS_TIME)

    moap_union = collect_union_objs(
        outputs_dir,
        args.methods,
        moap_allowed,
        args,
        moap_result.find_run_files,
        parse_moap_meta,
        moap_result._in_range,
        "MOAP union",
    )
    mokp_union = collect_union_objs(
        outputs_dir,
        args.methods,
        mokp_allowed,
        args,
        mokp_result.find_run_files,
        parse_mokp_meta,
        mokp_result._in_range,
        "MOKP union",
    )

    moap_points = collect_instance_points(
        outputs_dir,
        args.methods,
        moap_allowed,
        args,
        moap_result.find_run_files,
        parse_moap_meta,
        moap_result._in_range,
        "MOAP ref/ideal",
    )
    mokp_points = collect_instance_points(
        outputs_dir,
        args.methods,
        mokp_allowed,
        args,
        mokp_result.find_run_files,
        parse_mokp_meta,
        mokp_result._in_range,
        "MOKP ref/ideal",
    )

    moap_problem = {
        "name": "moap",
        "size_field": "n_agents",
        "size_label": lambda n_objs, size: f"objs-{n_objs}_agents-{size}",
        "hv_lib": args.hv_lib,
    }
    mokp_problem = {
        "name": "mokp",
        "size_field": "n_items",
        "size_label": lambda n_objs, size: f"objs-{n_objs}_items-{size}",
        "hv_lib": args.hv_lib,
    }

    moap_summary, moap_instances, moap_counts = compute_oracle_rows(
        moap_problem,
        moap_sizes,
        moap_union,
        moap_points,
        "n_agents",
    )
    mokp_summary, mokp_instances, mokp_counts = compute_oracle_rows(
        mokp_problem,
        mokp_sizes,
        mokp_union,
        mokp_points,
        "n_items",
    )

    if moap_summary:
        report_problem(moap_problem, moap_counts, moap_summary)
    else:
        print("MOAP: no oracle rows produced.")

    if mokp_summary:
        report_problem(mokp_problem, mokp_counts, mokp_summary)
    else:
        print("MOKP: no oracle rows produced.")

    moap_csv = (
        Path(args.save_moap_csv).expanduser().resolve()
        if args.save_moap_csv
        else results_dir / "moap_oracle_result.csv"
    )
    mokp_csv = (
        Path(args.save_mokp_csv).expanduser().resolve()
        if args.save_mokp_csv
        else results_dir / "mokp_oracle_result.csv"
    )
    moap_instance_csv = (
        Path(args.save_moap_instance_csv).expanduser().resolve()
        if args.save_moap_instance_csv
        else results_dir / "moap_oracle_instance_result.csv"
    )
    mokp_instance_csv = (
        Path(args.save_mokp_instance_csv).expanduser().resolve()
        if args.save_mokp_instance_csv
        else results_dir / "mokp_oracle_instance_result.csv"
    )

    if moap_summary:
        write_oracle_summary_moap(moap_csv, moap_summary)
        print(f"Wrote MOAP oracle CSV: {moap_csv}")
    if mokp_summary:
        write_oracle_summary_mokp(mokp_csv, mokp_summary)
        print(f"Wrote MOKP oracle CSV: {mokp_csv}")

    if moap_instances:
        write_instance_csv(moap_instance_csv, moap_instances, "n_agents")
        print(f"Wrote MOAP per-instance CSV: {moap_instance_csv}")
    if mokp_instances:
        write_instance_csv(mokp_instance_csv, mokp_instances, "n_items")
        print(f"Wrote MOKP per-instance CSV: {mokp_instance_csv}")


if __name__ == "__main__":
    main()
