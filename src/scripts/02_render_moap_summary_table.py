#!/usr/bin/env python3
"""Convert `moap_result.csv` into a LaTeX table.

Expected input: CSV produced by `src/scripts/01_summarize_moap_runs.py`.
"""

import argparse
import csv
import re
from collections import defaultdict
from math import isclose
from pathlib import Path

N_INIT_RE = re.compile(r"(?:^|/)n_init-(\d+)(?:/|$)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert moap_result.csv into a LaTeX table."
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to moap_result.csv (default: results/moap_result.csv).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Path to write LaTeX (default: results/moap_result.tex).",
    )
    parser.add_argument(
        "--caption",
        default="Results on the MOAP across all problem sizes.",
        help="LaTeX caption text.",
    )
    parser.add_argument(
        "--label",
        default="tab:moap_all_sizes",
        help="LaTeX label (without surrounding braces).",
    )
    parser.add_argument(
        "--table-position",
        default="t",
        help="LaTeX table position, e.g., t, ht, H.",
    )
    parser.add_argument(
        "--resize-width",
        default="0.85\\linewidth",
        help="Width argument passed to \\resizebox (set empty to disable).",
    )
    return parser.parse_args()


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def parse_number(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def round_to_int(value):
    number = parse_number(value)
    if number is None:
        return "--"
    if number >= 0:
        return str(int(number + 0.5))
    return str(int(number - 0.5))


def format_float(value, digits):
    number = parse_number(value)
    if number is None:
        return "--"
    return f"{number:.{digits}f}"


def format_hv(value, digits=2, scale=100.0):
    number = parse_number(value)
    if number is None:
        return "--"
    return f"{number * scale:.{digits}f}"


def parse_n_init(config):
    if not config:
        return None
    match = N_INIT_RE.search(str(config))
    if match:
        return match.group(1)
    return None


def escape_latex(text):
    text = str(text)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


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


def format_int_or_missing(value):
    number = parse_number(value)
    if number is None:
        return "--"
    if number < 0:
        return "--"
    return round_to_int(number)


def group_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (int(row.get("n_objs", 0)), int(row.get("n_agents", 0)))
        grouped[key].append(row)

    method_order = {"ksa": 0, "ea": 1, "aug_cheby": 2, "bpo": 3}

    def row_sort_key(row):
        method = str(row.get("method", "")).strip().lower()
        order = method_order.get(method, 99)
        n_init = parse_n_init(row.get("config"))
        n_init_key = int(n_init) if (method == "bpo" and n_init and n_init.isdigit()) else 10**9
        return (order, n_init_key, str(row.get("config", "")))

    ordered_groups = []
    for key in sorted(grouped.keys()):
        ordered_groups.append((key, sorted(grouped[key], key=row_sort_key)))
    return ordered_groups


def format_mean_hv(value, best_value):
    text = format_hv(value, digits=2, scale=100.0)
    number = parse_number(value)
    if number is None or best_value is None:
        return text
    if isclose(number, best_value, rel_tol=0.0, abs_tol=1e-12):
        return f"\\textbf{{{text}}}"
    return text


def latex_tabular(rows, title=None, highlight_best=True):
    tabular_cols = "rrlrrrrrr"
    lines = [
        f"\\begin{{tabular}}{{{tabular_cols}}}",
        "\\toprule",
    ]
    if title:
        lines += [
            f"\\multicolumn{{9}}{{c}}{{{escape_latex(title)}}}\\\\",
            "\\midrule",
        ]
    lines += [
        " &  &  & \\multicolumn{4}{c}{Hypervolume} &  &  \\\\",
        "\\cmidrule(lr){4-7}",
        "$m$ & $n$ & Method & Mean. & Std. & Min. & Max. "
        "& Iter. & $|\\hat{\\mathcal{Y}}_N|$ \\\\",
        "\\midrule",
    ]

    groups = group_rows(rows)
    for group_idx, ((n_objs, n_agents), group_rows_) in enumerate(groups):
        span = len(group_rows_)
        best_mean = None
        if highlight_best:
            means = [parse_number(row.get("mean_hv")) for row in group_rows_]
            means = [value for value in means if value is not None]
            if means:
                best_mean = max(means)
        for row_idx, row in enumerate(group_rows_):
            prefix = []
            if row_idx == 0:
                prefix = [
                    f"\\multirow{{{span}}}{{*}}{{{n_objs}}}",
                    f"\\multirow{{{span}}}{{*}}{{{n_agents}}}",
                ]
            else:
                prefix = ["", ""]
            line = " & ".join(
                prefix
                + [
                    escape_latex(method_label(row.get("method", ""), row.get("config"))),
                    (
                        format_mean_hv(row.get("mean_hv"), best_mean)
                        if highlight_best
                        else format_hv(row.get("mean_hv"), digits=2, scale=100.0)
                    ),
                    format_hv(row.get("std_hv"), digits=2, scale=100.0),
                    format_hv(row.get("min_hv"), digits=2, scale=100.0),
                    format_hv(row.get("max_hv"), digits=2, scale=100.0),
                    format_int_or_missing(row.get("mean_n_evals")),
                    format_int_or_missing(row.get("mean_n_nd")),
                ]
            )
            lines.append(line + " \\\\")

        if group_idx != len(groups) - 1:
            lines.append("\\midrule")

    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def latex_table(rows, caption, label, table_position, resize_width):
    lines = [
        f"\\begin{{table}}[{table_position}]",
        f"\\caption{{{caption}}}",
        "\\centering",
        "\\footnotesize",
    ]

    if resize_width:
        lines.append(f"\\resizebox{{{resize_width}}}{{!}}{{%")

    tabular = latex_tabular(rows).splitlines()
    lines += tabular
    if resize_width:
        lines[-1] += "}"
    lines += [f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = (
        Path(args.csv).expanduser().resolve()
        if args.csv
        else results_dir / "moap_result.csv"
    )
    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else results_dir / "moap_result.tex"
    )

    rows = read_csv(csv_path)
    out_path.write_text(
        latex_table(
            rows,
            caption=args.caption,
            label=args.label,
            table_position=args.table_position,
            resize_width=args.resize_width,
        ),
        encoding="utf-8",
    )
    print(f"Wrote LaTeX: {out_path}")


if __name__ == "__main__":
    main()
