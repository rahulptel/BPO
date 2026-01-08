#!/usr/bin/env python3
"""Create LaTeX tables summarizing final hypervolumes across solvers."""

import json
from pathlib import Path
from statistics import mean

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs"
RESULTS_DIR = REPO_ROOT / "results"


def parse_run_hyperparams(path):
    params = {}
    for part in path.parts:
        if part.startswith("batch_size_q-"):
            params["batch_size_q"] = part.split("batch_size_q-")[-1]
        elif part.startswith("sequential-"):
            params["sequential"] = part.split("sequential-")[-1]
    return params


def parse_iseed(path):
    for part in path.parts:
        if part.startswith("mokp-items-") and "_objs-" in part and "_iseed-" in part:
            try:
                return int(part.split("_iseed-")[-1])
            except ValueError:
                return None
    return None


def filter_run_paths(run_paths, required_params):
    if not required_params:
        return run_paths
    filtered = []
    for path in run_paths:
        params = parse_run_hyperparams(path)
        if all(params.get(key) == str(val) for key, val in required_params.items()):
            filtered.append(path)
    return filtered


def extract_last_hv_from_records(records):
    if not isinstance(records, list):
        return None
    for record in reversed(records):
        if isinstance(record, dict):
            hv = record.get("hv")
            if hv is not None:
                return hv
    return None


def extract_last_hv(payload):
    for key in ("hypervolume", "hv"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return value

    for key in ("final_record", "iterations", "iter_records"):
        hv = extract_last_hv_from_records(payload.get(key))
        if hv is not None:
            return hv
    return None


ALGORITHMS = {
    "bpo_q1": {
        "label": "BPO q=1",
        "root": OUTPUTS_DIR / "bpo",
        "file_glob": "run_bo_*.json",
        "extract": extract_last_hv,
        "filters": {"batch_size_q": "1"},
    },
    "aug_cheby": {
        "label": "AugCheby",
        "root": OUTPUTS_DIR / "aug_cheby",
        "file_glob": "run_aug_cheby_*.json",
        "extract": extract_last_hv,
    },
    "ksa": {
        "label": "KSA",
        "root": OUTPUTS_DIR / "ksa",
        "file_glob": "run_ksa_*.json",
        "extract": extract_last_hv,
        "nested_root": "ksa",
    },
    "ea": {
        "label": "EA",
        "root": OUTPUTS_DIR / "ea",
        "file_glob": "run_ea_*.json",
        "extract": extract_last_hv,
    },
}

# "bpo_q2_seq": {
#     "label": "BPO q=2 seq",
#     "root": OUTPUTS_DIR / "bpo",
#     "file_glob": "run_bo_*.json",
#     "extract": extract_last_hv,
#     "filters": {"batch_size_q": "2", "sequential": "True"},
# },
# "bpo_q2_joint": {
#     "label": "BPO q=2 joint",
#     "root": OUTPUTS_DIR / "bpo",
#     "file_glob": "run_bo_*.json",
#     "extract": extract_last_hv,
#     "filters": {"batch_size_q": "2", "sequential": "False"},
# },

N_VARS = (50, 250, 500)
N_OBJS = (3, 4, 5)
TABLE_PATH = RESULTS_DIR / "table.tex"


def collect_run_paths(root, n_vars, n_objs, file_glob):
    if not root.exists():
        return []
    pattern = f"mokp-items-{n_vars}_objs-{n_objs}_iseed-*"
    run_paths = []
    for instance_dir in sorted(root.glob(pattern)):
        iseed = parse_iseed(instance_dir)
        if iseed is not None and iseed > 10:
            continue
        run_paths.extend(instance_dir.rglob(file_glob))
    return run_paths


def resolve_root(cfg):
    root = cfg["root"]
    nested = cfg.get("nested_root")
    if nested:
        nested_root = root / nested
        if nested_root.exists():
            root = nested_root
    return root


def load_hypervolumes(paths, extractor):
    values = []
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        try:
            hv = extractor(payload)
        except (KeyError, IndexError, TypeError):
            continue
        if hv is not None:
            values.append(float(hv))
    return values


def format_hv(values):
    if not values:
        return "--"
    return f"{mean(values):.4f}"


def build_table_rows():
    rows = []
    for n_vars in N_VARS:
        for n_objs in N_OBJS:
            for algo_key, cfg in ALGORITHMS.items():
                root = resolve_root(cfg)
                paths = collect_run_paths(root, n_vars, n_objs, cfg["file_glob"])
                paths = filter_run_paths(paths, cfg.get("filters", {}))
                hvs = load_hypervolumes(paths, cfg["extract"])
                hv_text = format_hv(hvs)
                row = {
                    "n_vars": n_vars,
                    "n_objs": n_objs,
                    "algorithm": cfg["label"],
                    "runs": len(hvs),
                    "hv": hv_text,
                }
                rows.append(row)
    return rows


def render_table(rows):
    header = "\\begin{table}[ht]\n"
    header += "\\centering\n"
    header += "\\begin{tabular}{rrlrl}\n"
    header += "\\toprule\n"
    header += "\\multirow{2}{*}{n\\_vars} & \\multirow{2}{*}{n\\_objs} & \\multicolumn{3}{c}{Summary} \\\\\n"
    header += "\\cmidrule(lr){3-5}\n"
    header += " &  & algorithm & runs & hv \\\\\n"
    header += "\\midrule\n"

    body_lines = []
    for n_vars_idx, n_vars in enumerate(N_VARS):
        for n_objs in N_OBJS:
            group = [
                row
                for row in rows
                if row["n_vars"] == n_vars and row["n_objs"] == n_objs
            ]
            span = len(group)
            if span == 0:
                continue
            for idx, row in enumerate(group):
                if idx == 0:
                    line = (
                        f"\\multirow{{{span}}}{{*}}{{{n_vars}}} & "
                        f"\\multirow{{{span}}}{{*}}{{{n_objs}}} & "
                        f"{row['algorithm']} & {row['runs']} & {row['hv']} \\\\"
                    )
                else:
                    line = (
                        f" &  & {row['algorithm']} & {row['runs']} & {row['hv']} \\\\"
                    )
                body_lines.append(line)
            body_lines.append("\\cmidrule(lr){3-5}")
        if n_vars_idx < len(N_VARS) - 1:
            body_lines.append("\\midrule")

    footer = "\n\\bottomrule\n\\end{tabular}\n"
    footer += "\\caption{Final hypervolume summary across solvers.}\n"
    footer += "\\label{tab:hv_summary}\n\\end{table}\n"

    return header + "\n".join(body_lines) + footer


def main():
    rows = build_table_rows()
    table_tex = render_table(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(table_tex, encoding="utf-8")
    print(f"Wrote table to {TABLE_PATH}")


if __name__ == "__main__":
    main()
