#!/usr/bin/env python3
"""Create a side-by-side parallel vs oracle LaTeX table for MOAP and MOKP."""

import argparse
import csv
from math import isclose
from pathlib import Path

METHODS = ["ATS", "BPO-100", "BPO-150", "Oracle"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a side-by-side parallel vs oracle LaTeX table."
    )
    parser.add_argument(
        "--moap-parallel-csv",
        default=None,
        help="Path to moap_parallel_result.csv (default: results/moap_parallel_result.csv).",
    )
    parser.add_argument(
        "--mokp-parallel-csv",
        default=None,
        help="Path to mokp_parallel_result.csv (default: results/mokp_parallel_result.csv).",
    )
    parser.add_argument(
        "--moap-oracle-csv",
        default=None,
        help="Path to moap_oracle_result.csv (default: results/moap_oracle_result.csv).",
    )
    parser.add_argument(
        "--mokp-oracle-csv",
        default=None,
        help="Path to mokp_oracle_result.csv (default: results/mokp_oracle_result.csv).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Path to write LaTeX (default: results/parallel_result_latex.tex).",
    )
    parser.add_argument(
        "--caption",
        default="Parallel (k=5) HV results on MOKP (left) and MOAP (right) with Oracle.",
        help="LaTeX caption text.",
    )
    parser.add_argument(
        "--label",
        default="tab:hv_parallel_oracle",
        help="LaTeX label (without surrounding braces).",
    )
    parser.add_argument(
        "--table-position",
        default="tbp!",
        help="LaTeX table position, e.g., t, ht, H.",
    )
    parser.add_argument(
        "--table-star",
        action="store_true",
        help="Use table* environment (useful for two-column papers).",
    )
    parser.add_argument(
        "--minipage-width-left",
        default="0.49\\linewidth",
        help="Width for the left minipage.",
    )
    parser.add_argument(
        "--minipage-width-right",
        default="0.487\\linewidth",
        help="Width for the right minipage.",
    )
    parser.add_argument(
        "--resize-width",
        default="\\linewidth",
        help="Width argument passed to \\resizebox inside each minipage (set empty to disable).",
    )
    parser.add_argument(
        "--n-parallel",
        type=int,
        default=5,
        help="Number of parallel seeds to report (default: 5).",
    )
    parser.add_argument(
        "--no-highlight",
        action="store_true",
        help="Disable bolding the maximum mean HV per size.",
    )
    return parser.parse_args()


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def parse_int(value):
    number = parse_number(value)
    if number is None:
        return None
    return int(number)


def round_to_int(value):
    number = parse_number(value)
    if number is None:
        return "--"
    if number >= 0:
        return str(int(number + 0.5))
    return str(int(number - 0.5))


def format_hv(value, digits=2, scale=100.0):
    number = parse_number(value)
    if number is None:
        return "--"
    return f"{number * scale:.{digits}f}"


def format_int_or_missing(value):
    number = parse_number(value)
    if number is None or number < 0:
        return "--"
    return round_to_int(number)


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


def format_mean_hv(value, target_value=None, highlight=False, color=None):
    text = format_hv(value, digits=2, scale=100.0)
    if not highlight:
        return text
    if color:
        return f"\\textbf{{\\textcolor{{{color}}}{{{text}}}}}"
    return f"\\textbf{{{text}}}"


def filter_parallel_rows(rows, n_parallel, dim_key):
    filtered = []
    for row in rows:
        method = str(row.get("method", "")).strip()
        if method not in METHODS[:-1]:
            continue
        row_parallel = parse_int(row.get("n_parallel"))
        if row_parallel != n_parallel:
            continue
        n_objs = parse_int(row.get("n_objs"))
        dim_value = parse_int(row.get(dim_key))
        if n_objs is None or dim_value is None:
            continue
        filtered.append(
            {
                "n_objs": n_objs,
                dim_key: dim_value,
                "method": method,
                "mean_hv": parse_number(row.get("hv")),
                "std_hv": parse_number(row.get("std_hv")),
                "min_hv": parse_number(row.get("min_hv")),
                "max_hv": parse_number(row.get("max_hv")),
                "mean_n_nd": parse_number(row.get("mean_n_nd")),
            }
        )
    return filtered


def build_oracle_map(rows, dim_key):
    oracle_map = {}
    for row in rows:
        method = str(row.get("method", "")).strip().lower()
        if method != "oracle":
            continue
        n_objs = parse_int(row.get("n_objs"))
        dim_value = parse_int(row.get(dim_key))
        if n_objs is None or dim_value is None:
            continue
        key = (n_objs, dim_value)
        oracle_map[key] = {
            "n_objs": n_objs,
            dim_key: dim_value,
            "method": "Oracle",
            "mean_hv": parse_number(row.get("mean_hv")),
            "std_hv": parse_number(row.get("std_hv")),
            "min_hv": parse_number(row.get("min_hv")),
            "max_hv": parse_number(row.get("max_hv")),
            "mean_n_nd": parse_number(row.get("mean_n_nd")),
        }
    return oracle_map


def build_groups(parallel_rows, oracle_rows, dim_key, size_label):
    parallel_map = {}
    for row in parallel_rows:
        key = (row["n_objs"], row[dim_key])
        parallel_map.setdefault(key, {})[row["method"]] = row

    oracle_map = build_oracle_map(oracle_rows, dim_key)

    size_keys = sorted(set(parallel_map.keys()) | set(oracle_map.keys()))
    groups = []
    for n_objs, dim_value in size_keys:
        group_rows = []
        for method in METHODS:
            if method == "Oracle":
                row = oracle_map.get((n_objs, dim_value))
            else:
                row = parallel_map.get((n_objs, dim_value), {}).get(method)
            if row is None:
                print(
                    f"Missing {method} for {size_label} size m={n_objs}, n={dim_value}"
                )
                row = {
                    "n_objs": n_objs,
                    dim_key: dim_value,
                    "method": method,
                    "mean_hv": None,
                    "std_hv": None,
                    "min_hv": None,
                    "max_hv": None,
                    "mean_n_nd": None,
                }
            group_rows.append(row)
        groups.append((n_objs, dim_value, group_rows))
    return groups


def latex_tabular(groups, dim_symbol, title=None, highlight_best=True):
    tabular_cols = "rrlrrrrr"
    lines = [
        f"\\begin{{tabular}}{{{tabular_cols}}}",
        "\\toprule",
    ]
    if title:
        lines += [
            f"\\multicolumn{{8}}{{c}}{{{escape_latex(title)}}}\\\\",
            "\\midrule",
        ]
    lines += [
        " &  &  & \\multicolumn{4}{c}{Hypervolume} &  \\\\",
        "\\cmidrule(lr){4-7}",
        f"$m$ & {dim_symbol} & Method & Mean. & Std. & Min. & Max. "
        "& $|\\hat{\\mathcal{Y}}_N|$ \\\\",
        "\\midrule",
    ]

    cmidrule_end = len(tabular_cols)
    for group_idx, (n_objs, dim_value, group_rows) in enumerate(groups):
        span = len(group_rows)
        oracle_mean = None
        if highlight_best:
            for row in group_rows:
                if row.get("method") == "Oracle":
                    oracle_mean = parse_number(row.get("mean_hv"))
                    break
        closest_method = None
        if highlight_best and oracle_mean is not None:
            candidates = []
            for row in group_rows:
                if row.get("method") == "Oracle":
                    continue
                mean_value = parse_number(row.get("mean_hv"))
                if mean_value is None:
                    continue
                candidates.append((abs(mean_value - oracle_mean), row.get("method")))
            if candidates:
                candidates.sort()
                closest_method = candidates[0][1]

        for row_idx, row in enumerate(group_rows):
            method = row.get("method", "")
            if method == "Oracle":
                lines.append(f"\\cmidrule(lr){{2-{cmidrule_end}}}")
            prefix = []
            if row_idx == 0:
                prefix = [
                    f"\\multirow{{{span}}}{{*}}{{{n_objs}}}",
                    f"\\multirow{{{span}}}{{*}}{{{dim_value}}}",
                ]
            else:
                prefix = ["", ""]

            method_text = escape_latex(method)
            if highlight_best and method == "Oracle":
                method_text = f"\\textbf{{\\textcolor{{red}}{{{method_text}}}}}"
            elif highlight_best and method == closest_method:
                method_text = f"\\textbf{{{method_text}}}"

            line = " & ".join(
                prefix
                + [
                    method_text,
                    format_mean_hv(
                        row.get("mean_hv"),
                        highlight=highlight_best and method == "Oracle",
                        color="red",
                    )
                    if highlight_best and method == "Oracle"
                    else format_mean_hv(
                        row.get("mean_hv"),
                        highlight=highlight_best and method == closest_method,
                    ),
                    format_hv(row.get("std_hv"), digits=2, scale=100.0),
                    format_hv(row.get("min_hv"), digits=2, scale=100.0),
                    format_hv(row.get("max_hv"), digits=2, scale=100.0),
                    format_int_or_missing(row.get("mean_n_nd")),
                ]
            )
            lines.append(line + " \\\\")

        if group_idx != len(groups) - 1:
            lines.append("\\midrule")

    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def wrap_resizebox(tabular_lines, resize_width):
    if not resize_width:
        return tabular_lines
    lines = [f"\\resizebox{{{resize_width}}}{{!}}{{%"]
    lines += tabular_lines
    lines[-1] += "}"
    return lines


def side_by_side_table(
    left_tabular,
    right_tabular,
    caption,
    label,
    table_position,
    table_star,
    minipage_width_left,
    minipage_width_right,
    resize_width,
):
    env = "table*" if table_star else "table"
    lines = [
        f"\\begin{{{env}}}[{table_position}]",
        f"\\caption{{{caption}}}",
        "\\centering",
        "\\footnotesize",
        f"\\begin{{minipage}}[t]{{{minipage_width_left}}}",
        "\\centering",
    ]

    lines += wrap_resizebox(left_tabular.splitlines(), resize_width)
    lines += [
        "\\end{minipage}\\hfill",
        f"\\begin{{minipage}}[t]{{{minipage_width_right}}}",
        "\\centering",
    ]
    lines += wrap_resizebox(right_tabular.splitlines(), resize_width)
    lines += [
        "\\end{minipage}",
        f"\\label{{{label}}}",
        f"\\end{{{env}}}",
    ]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    moap_parallel_csv = (
        Path(args.moap_parallel_csv).expanduser().resolve()
        if args.moap_parallel_csv
        else results_dir / "moap_parallel_result.csv"
    )
    mokp_parallel_csv = (
        Path(args.mokp_parallel_csv).expanduser().resolve()
        if args.mokp_parallel_csv
        else results_dir / "mokp_parallel_result.csv"
    )
    moap_oracle_csv = (
        Path(args.moap_oracle_csv).expanduser().resolve()
        if args.moap_oracle_csv
        else results_dir / "moap_oracle_result.csv"
    )
    mokp_oracle_csv = (
        Path(args.mokp_oracle_csv).expanduser().resolve()
        if args.mokp_oracle_csv
        else results_dir / "mokp_oracle_result.csv"
    )
    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else results_dir / "parallel_result_latex.tex"
    )

    moap_parallel_rows = filter_parallel_rows(
        read_csv(moap_parallel_csv), args.n_parallel, "n_agents"
    )
    mokp_parallel_rows = filter_parallel_rows(
        read_csv(mokp_parallel_csv), args.n_parallel, "n_items"
    )
    moap_oracle_rows = read_csv(moap_oracle_csv)
    mokp_oracle_rows = read_csv(mokp_oracle_csv)

    highlight_best = not args.no_highlight
    moap_groups = build_groups(
        moap_parallel_rows, moap_oracle_rows, "n_agents", "MOAP"
    )
    mokp_groups = build_groups(
        mokp_parallel_rows, mokp_oracle_rows, "n_items", "MOKP"
    )

    tex = side_by_side_table(
        left_tabular=latex_tabular(
            mokp_groups, "$n$", title="MOKP", highlight_best=highlight_best
        ),
        right_tabular=latex_tabular(
            moap_groups, "$n$", title="MOAP", highlight_best=highlight_best
        ),
        caption=args.caption,
        label=args.label,
        table_position=args.table_position,
        table_star=args.table_star,
        minipage_width_left=args.minipage_width_left,
        minipage_width_right=args.minipage_width_right,
        resize_width=args.resize_width,
    )
    out_path.write_text(tex, encoding="utf-8")
    print(f"Wrote LaTeX: {out_path}")


if __name__ == "__main__":
    main()
