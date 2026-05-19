#!/usr/bin/env python3
"""Create a side-by-side MOAP + MOKP LaTeX table from CSV summaries."""

import argparse
import importlib.util
from pathlib import Path


def load_script_module(name, filename):
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


moap = load_script_module("moap_summary_table", "02_render_moap_summary_table.py")
mokp = load_script_module("mokp_summary_table", "02_render_mokp_summary_table.py")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a side-by-side MOAP + MOKP LaTeX table."
    )
    parser.add_argument(
        "--moap-csv",
        default=None,
        help="Path to moap_result.csv (default: results/moap_result.csv).",
    )
    parser.add_argument(
        "--mokp-csv",
        default=None,
        help="Path to mokp_result.csv (default: results/mokp_result.csv).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Path to write LaTeX (default: results/moap_mokp_side_by_side.tex).",
    )
    parser.add_argument(
        "--caption",
        default="Final HV results on MOKP (left) and MOAP (right) across all problem sizes.",
        help="LaTeX caption text.",
    )
    parser.add_argument(
        "--label",
        default="tab:hv_final",
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
        "--left-title",
        default="MOKP",
        help="Title shown inside the left table.",
    )
    parser.add_argument(
        "--right-title",
        default="MOAP",
        help="Title shown inside the right table.",
    )
    parser.add_argument(
        "--no-highlight",
        action="store_true",
        help="Disable bolding the maximum mean HV per size.",
    )
    return parser.parse_args()


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
    moap_csv = (
        Path(args.moap_csv).expanduser().resolve()
        if args.moap_csv
        else results_dir / "moap_result.csv"
    )
    mokp_csv = (
        Path(args.mokp_csv).expanduser().resolve()
        if args.mokp_csv
        else results_dir / "mokp_result.csv"
    )
    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else results_dir / "moap_mokp_side_by_side.tex"
    )

    moap_rows = moap.read_csv(moap_csv)
    mokp_rows = mokp.read_csv(mokp_csv)
    highlight_best = not args.no_highlight
    tex = side_by_side_table(
        left_tabular=mokp.latex_tabular(
            mokp_rows, title=args.left_title, highlight_best=highlight_best
        ),
        right_tabular=moap.latex_tabular(
            moap_rows, title=args.right_title, highlight_best=highlight_best
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
