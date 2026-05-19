#!/usr/bin/env python3
"""Summarize final MOAP hypervolume results from the outputs directory."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

MOAP_PART_RE = re.compile(r"moap-agents-(\d+)_objs-(\d+)_iseed-(\d+)")
MOAP_OBJ_AGENT_TIME = [
    (3, 175, 240),
    (3, 200, 480),
    (4, 100, 240),
    (4, 125, 480),
    (5, 75, 480),
    (6, 50, 240),
    (7, 50, 480),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect final MOAP hypervolume results from outputs."
    )
    parser.add_argument(
        "--outputs-dir",
        default=None,
        help="Path to raw outputs/ directory (default: repository outputs/).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["ksa", "ea", "aug_cheby", "bpo"],
        help="Methods to include (default: aug_cheby bpo ksa ea).",
    )
    parser.add_argument(
        "--iseed-min",
        type=int,
        default=26,
        help="Minimum iseed to include (default: 25).",
    )
    parser.add_argument(
        "--iseed-max",
        type=int,
        default=50,
        help="Maximum iseed to include (default: 50).",
    )
    parser.add_argument(
        "--rseed-min",
        type=int,
        default=1,
        help="Minimum rseed to include (default: 1).",
    )
    parser.add_argument(
        "--rseed-max",
        type=int,
        default=5,
        help="Maximum rseed to include (default: 5).",
    )
    parser.add_argument(
        "--save-csv",
        default=None,
        help="Optional path to write a CSV summary (default: results/moap_result.csv).",
    )
    parser.add_argument(
        "--save-run-csv",
        default=None,
        help="Optional path to write per-run MOAP results (default: results/moap_result_runs.csv).",
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


def extract_final_hv(payload):
    final_record = payload.get("final_record")

    if isinstance(final_record, dict):
        hv = final_record.get("hv")
        if isinstance(hv, (int, float)):
            return float(hv)

    if isinstance(final_record, list):
        for record in reversed(final_record):
            if not isinstance(record, dict):
                continue
            hv = record.get("hv")
            if isinstance(hv, (int, float)):
                return float(hv)

    for key in ("hv", "hypervolume"):
        hv = payload.get(key)
        if isinstance(hv, (int, float)):
            return float(hv)

    iter_records = payload.get("iter_records")
    if iter_records is None:
        iter_records = payload.get("iterations")
    if isinstance(iter_records, list):
        for record in reversed(iter_records):
            if not isinstance(record, dict):
                continue
            hv = record.get("hv")
            if isinstance(hv, (int, float)):
                return float(hv)

    return None


def extract_final_n_nd(payload):
    final_record = payload.get("final_record")
    if isinstance(final_record, dict):
        value = final_record.get("n_nd")
        if isinstance(value, (int, float)):
            return float(value)
    if isinstance(final_record, list):
        for record in reversed(final_record):
            if not isinstance(record, dict):
                continue
            value = record.get("n_nd")
            if isinstance(value, (int, float)):
                return float(value)

    iter_records = payload.get("iter_records")
    if iter_records is None:
        iter_records = payload.get("iterations")
    if isinstance(iter_records, list):
        for record in reversed(iter_records):
            if not isinstance(record, dict):
                continue
            value = record.get("n_nd")
            if isinstance(value, (int, float)):
                return float(value)
    return None


def extract_n_evaluations(method, payload):
    if method == "ea":
        return -1

    if method == "ksa":
        n_evaluations = payload.get("n_evaluations")
        if isinstance(n_evaluations, int):
            return n_evaluations

    if method == "bpo" or method == "aug_cheby":
        objs = payload.get("objs")
        if objs is not None and isinstance(objs, list):
            return len(objs)
        if "n_evaluations" in payload:
            n_evaluations = payload.get("n_evaluations")
            if isinstance(n_evaluations, int):
                return n_evaluations

    return -1


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


def summarize_group(hvs):
    if not hvs:
        return None
    mu = mean(hvs)
    if len(hvs) < 2:
        sigma = 0.0
    else:
        mu2 = mean([(x - mu) ** 2 for x in hvs])
        sigma = mu2**0.5
    return mu, sigma, min(hvs), max(hvs)


def mean_or_none(values):
    if not values:
        return None
    return float(mean(values))


def _in_range(value, low, high):
    if value is None:
        return False
    if low is not None and value < low:
        return False
    if high is not None and value > high:
        return False
    return True


def write_csv(path, rows):
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
                    _csv_escape(row.get("config", "")),
                    str(row.get("runs", "")),
                    str(row.get("unique_iseeds", "")),
                    str(row.get("unique_rseeds", "")),
                    _csv_number(row.get("mean_n_evals")),
                    _csv_number(row.get("min_n_evals")),
                    _csv_number(row.get("max_n_evals")),
                    _csv_number(row.get("mean_n_nd")),
                    f"{row.get('mean_hv', 0.0):.10f}",
                    f"{row.get('std_hv', 0.0):.10f}",
                    f"{row.get('min_hv', 0.0):.10f}",
                    f"{row.get('max_hv', 0.0):.10f}",
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_csv(path, rows):
    header = [
        "problem",
        "n_agents",
        "n_objs",
        "iseed",
        "rseed",
        "method",
        "config",
        "final_hv",
    ]
    lines = [",".join(header)]
    for row in rows:
        lines.append(
            ",".join(
                [
                    str(row.get("problem", "")),
                    str(row.get("n_agents", "")),
                    str(row.get("n_objs", "")),
                    str(row.get("iseed", "")),
                    str(row.get("rseed", "")),
                    str(row.get("method", "")),
                    _csv_escape(row.get("config", "")),
                    f"{row.get('final_hv', 0.0):.10f}",
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _csv_escape(value):
    text = str(value)
    if "," in text or '"' in text or "\n" in text:
        text = text.replace('"', '""')
        return f'"{text}"'
    return text


def _csv_number(value):
    if value is None:
        return ""
    return f"{float(value):.6f}"


def _format_number(value, digits):
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


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

    allowed_specs = defaultdict(set)
    for n_objs, n_agents, time_limit in MOAP_OBJ_AGENT_TIME:
        allowed_specs[(n_objs, n_agents)].add(int(time_limit))
    expected_iseeds = None
    if args.iseed_min is not None and args.iseed_max is not None:
        expected_iseeds = set(range(int(args.iseed_min), int(args.iseed_max) + 1))
    expected_rseeds = None
    if args.rseed_min is not None and args.rseed_max is not None:
        expected_rseeds = set(range(int(args.rseed_min), int(args.rseed_max) + 1))

    rows = []
    run_rows = []
    for method in args.methods:
        method_root, run_files = find_run_files(outputs_dir, method)

        grouped = defaultdict(
            lambda: {
                "hvs": [],
                "n_evals": [],
                "n_nd": [],
                "iseeds": set(),
                "rseeds": set(),
                "n_agents": None,
            }
        )
        missing_hv = 0
        unmatched = 0

        for run_path in run_files:
            desc = _parse_moap_descriptor_from_path(run_path)
            if desc is None:
                unmatched += 1
                continue
            time_limit = _parse_time_limit_from_path(run_path)
            if time_limit is None:
                continue
            allowed_times = allowed_specs.get((desc["objs"], desc["agents"]))
            if not allowed_times or time_limit not in allowed_times:
                continue
            if not _in_range(desc["iseed"], args.iseed_min, args.iseed_max):
                continue
            rseed = _parse_rseed_from_path(run_path)
            if (
                rseed is not None
                and (args.rseed_min is not None or args.rseed_max is not None)
                and not _in_range(rseed, args.rseed_min, args.rseed_max)
            ):
                continue

            try:
                payload = load_json(run_path)
            except (OSError, json.JSONDecodeError):
                continue
            hv = extract_final_hv(payload)

            if hv is None:
                missing_hv += 1
                continue
            n_nd = extract_final_n_nd(payload)
            n_evals = extract_n_evaluations(method, payload)

            cfg_sig = config_signature(method_root, run_path, desc["instance_part"])
            if method in {"aug_cheby", "bpo"}:
                run_rows.append(
                    {
                        "problem": "moap",
                        "n_agents": desc["agents"],
                        "n_objs": desc["objs"],
                        "iseed": desc["iseed"],
                        "rseed": rseed if rseed is not None else "",
                        "method": method,
                        "config": cfg_sig,
                        "final_hv": float(hv),
                    }
                )

            key = (desc["objs"], desc["agents"], cfg_sig)
            bucket = grouped[key]
            bucket["hvs"].append(float(hv))
            if n_nd is not None:
                bucket["n_nd"].append(float(n_nd))
            if n_evals is not None:
                bucket["n_evals"].append(int(n_evals))
            bucket["iseeds"].add(int(desc["iseed"]))
            if rseed is not None:
                bucket["rseeds"].add(int(rseed))

        if not grouped:
            continue

        for n_objs, n_agents, cfg_sig in grouped.keys():
            data = grouped[(n_objs, n_agents, cfg_sig)]
            stats = summarize_group(data["hvs"])
            if stats is None:
                continue
            mu, sigma, lo, hi = stats
            mean_n_evals = mean_or_none(data["n_evals"])
            mean_n_nd = mean_or_none(data["n_nd"])
            missing_iseeds = []
            if expected_iseeds is not None:
                missing_iseeds = sorted(expected_iseeds - data["iseeds"])
            missing_rseeds = []
            if expected_rseeds is not None:
                missing_rseeds = sorted(expected_rseeds - data["rseeds"])

            rows.append(
                {
                    "method": method,
                    "n_objs": n_objs,
                    "n_agents": n_agents,
                    "config": cfg_sig,
                    "runs": len(data["hvs"]),
                    "unique_iseeds": len(data["iseeds"]),
                    "unique_rseeds": len(data["rseeds"]),
                    "mean_n_evals": mean_n_evals,
                    "min_n_evals": min(data["n_evals"]) if data["n_evals"] else None,
                    "max_n_evals": max(data["n_evals"]) if data["n_evals"] else None,
                    "mean_n_nd": mean_n_nd,
                    "mean_hv": mu,
                    "std_hv": sigma,
                    "min_hv": lo,
                    "max_hv": hi,
                    "missing_iseeds": missing_iseeds,
                    "missing_rseeds": missing_rseeds,
                }
            )

        if missing_hv:
            print(f"(skipped {missing_hv} {method} runs missing HV)")
        if unmatched:
            print(
                f"(skipped {unmatched} {method} json files not matching moap descriptor)"
            )

    if not rows:
        print("No matching runs found.")
    else:
        method_order = {"aug_cheby": 2, "bpo": 3, "ksa": 0, "ea": 1}
        rows = sorted(
            rows,
            key=lambda r: (
                r.get("n_objs", 0),
                r.get("n_agents", 0),
                method_order.get(r.get("method"), 99),
                r.get("method", ""),
                r.get("config", ""),
            ),
        )
        current_obj = None
        current_agents = None
        for row in rows:
            n_objs = row["n_objs"]
            n_agents = row["n_agents"]
            if (n_objs, n_agents) != (current_obj, current_agents):
                current_obj, current_agents = n_objs, n_agents
                print(f"\n-- objs={n_objs} agents={n_agents} --")
            print(
                f"- method={row['method']} config={row['config']}\n"
                f"  runs={row['runs']}  iseed={row['unique_iseeds']}  "
                f"rseed={row['unique_rseeds']}\n"
                f"  mean={row['mean_hv']:.6f}  std={row['std_hv']:.6f}  "
                f"min={row['min_hv']:.6f}  max={row['max_hv']:.6f}\n"
                f"  avg_evals={_format_number(row.get('mean_n_evals'), 2)}  "
                f"avg_n_nd={_format_number(row.get('mean_n_nd'), 2)}"
            )
            if row.get("missing_iseeds"):
                print(f"  missing_iseed={row['missing_iseeds']}")
            if row.get("missing_rseeds"):
                print(f"  missing_rseed={row['missing_rseeds']}")

    if args.save_csv:
        csv_path = Path(args.save_csv).expanduser().resolve()
    else:
        csv_path = results_dir / "moap_result.csv"
    if csv_path:
        write_csv(csv_path, rows)
        print(f"\nWrote CSV: {csv_path}")

    if args.save_run_csv:
        csv_path = Path(args.save_run_csv).expanduser().resolve()
    else:
        csv_path = results_dir / "moap_result_runs.csv"
    if csv_path:
        write_run_csv(csv_path, run_rows)
        print(f"\nWrote run CSV: {csv_path}")


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    main()
