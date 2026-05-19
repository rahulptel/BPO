#!/usr/bin/env python3
"""Generate MOAP/MOKP full plots and a combined 1x4 summary plot."""

import argparse
import csv
import importlib.util
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def load_script_module(name, filename):
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


moap_result = load_script_module("moap_result", "01_summarize_moap_runs.py")
mokp_result = load_script_module("mokp_result", "01_summarize_mokp_runs.py")
moap_latex = load_script_module("moap_latex", "02_render_moap_summary_table.py")
mokp_latex = load_script_module("mokp_latex", "02_render_mokp_summary_table.py")

METHOD_ORDER = {"aug_cheby": 1, "bpo": 2}
IGNORED_METHODS = {"ea"}
METHOD_COLORS = {
    "ea": "#F58518",
    "aug_cheby": "#54A24B",
    "bpo_100": "#E45756",
    "bpo_150": "#B279A2",
    "bpo": "#9D755D",
    "other": "#7F7F7F",
}

TITLE_FONTSIZE = 19
LABEL_FONTSIZE = 18
TICK_FONTSIZE = 16
LEGEND_FONTSIZE = 18
XLABEL_POSITIONS = {(1, 4), (2, 1), (2, 2), (2, 3)}
YLABEL_POSITIONS = {(1, 1), (2, 1)}

PROBLEM_CONFIG = {
    "moap": {
        "csv_name": "moap_parallel_result.csv",
        "out_name": "moap_parallel_result_plot.png",
        "size_key": "n_agents",
        "size_fallback_key": None,
        "size_title": "agents",
        "size_sequence": moap_result.MOAP_OBJ_AGENT_TIME,
        "method_label_fn": moap_latex.method_label,
        "parse_n_init_fn": moap_latex.parse_n_init,
        "problem_label": "MOAP",
    },
    "mokp": {
        "csv_name": "mokp_parallel_result.csv",
        "out_name": "mokp_parallel_result_plot.png",
        "size_key": "n_items",
        "size_fallback_key": "n_agents",
        "size_title": "items",
        "size_sequence": mokp_result.MOKP_OBJ_ITEMS_TIME,
        "method_label_fn": mokp_latex.method_label,
        "parse_n_init_fn": mokp_latex.parse_n_init,
        "problem_label": "MOKP",
    },
}

COMBINED_PANELS = [
    ("mokp", 3, 1250),
    ("mokp", 6, 500),
    ("moap", 3, 200),
    ("moap", 7, 50),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-problem parallel-result plots and an additional "
            "combined 1x4 summary plot."
        )
    )
    parser.add_argument(
        "--csv-moap", default=None, help="Path to moap_parallel_result.csv."
    )
    parser.add_argument(
        "--csv-mokp", default=None, help="Path to mokp_parallel_result.csv."
    )
    parser.add_argument("--out-moap", default=None, help="Output image path for MOAP.")
    parser.add_argument("--out-mokp", default=None, help="Output image path for MOKP.")
    parser.add_argument(
        "--out-combined",
        default=None,
        help="Output image path for the combined plot (default: combined_parallel_result_plot.png).",
    )
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=2,
        help="Number of subplot rows for per-problem plots (default: 2).",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=4,
        help="Number of subplot columns for per-problem plots (default: 4).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures after saving.",
    )
    return parser.parse_args()


def parse_int(value):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_float(value):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def read_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def size_order(size_sequence):
    order = []
    seen = set()
    for n_objs, size_val, _ in size_sequence:
        key = (int(n_objs), int(size_val))
        if key not in seen:
            seen.add(key)
            order.append(key)
    return order


def method_sort_key(method_raw, config, label, parse_n_init_fn):
    method = str(method_raw).strip().lower()
    rank = METHOD_ORDER.get(method, 99)
    if method == "bpo":
        n_init = parse_n_init_fn(config)
        n_init_rank = int(n_init) if n_init and n_init.isdigit() else 10**9
    else:
        n_init_rank = 10**9
    return (rank, n_init_rank, str(label))


def method_color(method_raw, config, parse_n_init_fn):
    method = str(method_raw).strip().lower()
    if method == "bpo":
        n_init = parse_n_init_fn(config)
        if n_init == "100":
            return METHOD_COLORS["bpo_100"]
        if n_init == "150":
            return METHOD_COLORS["bpo_150"]
        return METHOD_COLORS["bpo"]
    return METHOD_COLORS.get(method, METHOD_COLORS["other"])


def build_series(rows, size_key, size_fallback_key, method_label_fn):
    series_by_size = defaultdict(dict)
    x_values = set()

    for row in rows:
        n_objs = parse_int(row.get("n_objs"))
        size_val = parse_int(row.get(size_key))
        if size_val is None and size_fallback_key:
            size_val = parse_int(row.get(size_fallback_key))
        n_parallel = parse_int(row.get("n_parallel"))
        hv = parse_float(row.get("hv"))

        if n_objs is None or size_val is None or n_parallel is None or hv is None:
            continue

        method_raw = str(row.get("method_raw", "")).strip().lower()
        config = str(row.get("config", "")).strip()
        label = str(row.get("method", "")).strip()
        if not label:
            label = method_label_fn(method_raw, config)
        if method_raw in IGNORED_METHODS or label.strip().lower() == "nsga-ii":
            continue

        panel_key = (n_objs, size_val)
        method_key = (method_raw, config, label)
        if method_key not in series_by_size[panel_key]:
            series_by_size[panel_key][method_key] = {}
        series_by_size[panel_key][method_key][n_parallel] = hv
        x_values.add(n_parallel)

    return series_by_size, sorted(x_values)


def should_show_xlabel(row, col, grid_rows, grid_cols):
    if grid_rows == 2 and grid_cols == 4:
        return (row, col) in XLABEL_POSITIONS
    return row == grid_rows


def should_show_ylabel(row, col, grid_rows, grid_cols):
    if grid_rows == 2 and grid_cols == 4:
        return (row, col) in YLABEL_POSITIONS
    return col == 1


def plot_problem_grid(problem_name, cfg, data, args, out_path):
    ordered_sizes = data["ordered_sizes"]
    series_by_size = data["series_by_size"]
    x_values = data["x_values"]

    max_panels = args.grid_rows * args.grid_cols
    if len(ordered_sizes) > max_panels:
        raise ValueError(
            f"Grid {args.grid_rows}x{args.grid_cols} supports {max_panels} panels, "
            f"but found {len(ordered_sizes)} sizes for {problem_name}."
        )

    fig, axes = plt.subplots(
        args.grid_rows,
        args.grid_cols,
        figsize=(4.8 * args.grid_cols, 3.3 * args.grid_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()
    legend_handles = {}

    for panel_idx, ((n_objs, size_val), ax) in enumerate(
        zip(ordered_sizes, axes_flat), start=1
    ):
        row = ((panel_idx - 1) // args.grid_cols) + 1
        col = ((panel_idx - 1) % args.grid_cols) + 1

        ax.set_title(
            f"objs={n_objs}, {cfg['size_title']}={size_val}", fontsize=TITLE_FONTSIZE
        )
        if should_show_xlabel(row, col, args.grid_rows, args.grid_cols):
            ax.set_xlabel("Seed Count (k)", fontsize=LABEL_FONTSIZE)
        else:
            ax.set_xlabel("")
        if should_show_ylabel(row, col, args.grid_rows, args.grid_cols):
            ax.set_ylabel("Hypervolume %", fontsize=LABEL_FONTSIZE)
        else:
            ax.set_ylabel("")

        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
        ax.set_xticks(x_values)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:.2f}"))
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

        methods = series_by_size.get((n_objs, size_val), {})
        ordered_methods = sorted(
            methods.keys(),
            key=lambda key: method_sort_key(
                key[0], key[1], key[2], cfg["parse_n_init_fn"]
            ),
        )

        for method_raw, config, label in ordered_methods:
            points = methods[(method_raw, config, label)]
            xs = sorted(points.keys())
            ys = [points[x] for x in xs]
            (line,) = ax.plot(
                xs,
                ys,
                marker="o",
                markersize=5.6,
                linewidth=2.3,
                color=method_color(method_raw, config, cfg["parse_n_init_fn"]),
                label=label,
            )
            if label not in legend_handles:
                legend_handles[label] = line

        if not ordered_methods:
            ax.set_axis_off()

    for ax in axes_flat[len(ordered_sizes) :]:
        ax.set_axis_off()

    legend_row = min(max(args.grid_rows - 1, 0), args.grid_rows - 1)
    legend_col = min(max(args.grid_cols - 1, 0), args.grid_cols - 1)
    legend_ax = axes[legend_row][legend_col]
    if legend_handles:
        legend_ax.legend(
            list(legend_handles.values()),
            list(legend_handles.keys()),
            loc="center",
            fontsize=LEGEND_FONTSIZE,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"Saved figure: {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


def plot_combined_grid(problem_data, args, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(4.9 * 4, 4.4), squeeze=False)
    axes_row = axes[0]
    legend_handles = {}

    for idx, (problem_name, n_objs, size_val) in enumerate(COMBINED_PANELS):
        cfg = PROBLEM_CONFIG[problem_name]
        data = problem_data[problem_name]
        ax = axes_row[idx]
        methods = data["series_by_size"].get((n_objs, size_val), {})

        ax.set_title(
            f"{cfg['problem_label']} ({n_objs}, {size_val})",
            fontsize=TITLE_FONTSIZE,
        )
        ax.set_xlabel("Seed Count (k)", fontsize=LABEL_FONTSIZE)
        if idx == 0:
            ax.set_ylabel("Hypervolume %", fontsize=LABEL_FONTSIZE)
        else:
            ax.set_ylabel("")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
        ax.set_xticks(data["x_values"])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:.2f}"))
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

        ordered_methods = sorted(
            methods.keys(),
            key=lambda key: method_sort_key(
                key[0], key[1], key[2], cfg["parse_n_init_fn"]
            ),
        )
        for method_raw, config, label in ordered_methods:
            points = methods[(method_raw, config, label)]
            xs = sorted(points.keys())
            ys = [points[x] for x in xs]
            (line,) = ax.plot(
                xs,
                ys,
                marker="o",
                markersize=5.6,
                linewidth=2.3,
                color=method_color(method_raw, config, cfg["parse_n_init_fn"]),
                label=label,
            )
            if label not in legend_handles:
                legend_handles[label] = line

        if not ordered_methods:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )

    if legend_handles:
        fig.legend(
            list(legend_handles.values()),
            list(legend_handles.keys()),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=len(legend_handles),
            fontsize=LEGEND_FONTSIZE,
            frameon=True,
        )

    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"Saved figure: {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


def resolve_paths(problem_name, cfg, args, results_dir):
    csv_override = getattr(args, f"csv_{problem_name}")
    out_override = getattr(args, f"out_{problem_name}")
    csv_path = (
        Path(csv_override).expanduser().resolve()
        if csv_override
        else results_dir / cfg["csv_name"]
    )
    out_path = (
        Path(out_override).expanduser().resolve()
        if out_override
        else results_dir / cfg["out_name"]
    )
    return csv_path, out_path


def load_problem_data(problem_name, cfg, csv_path):
    rows = read_rows(csv_path)
    series_by_size, x_values = build_series(
        rows,
        cfg["size_key"],
        cfg["size_fallback_key"],
        cfg["method_label_fn"],
    )
    if not series_by_size:
        raise ValueError(f"No plottable rows found in {csv_path} for {problem_name}.")

    ordered_from_table = size_order(cfg["size_sequence"])
    ordered_sizes = [key for key in ordered_from_table if key in series_by_size]
    ordered_sizes += sorted(
        key for key in series_by_size if key not in set(ordered_from_table)
    )

    return {
        "series_by_size": series_by_size,
        "x_values": x_values,
        "ordered_sizes": ordered_sizes,
    }


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    problem_data = {}
    output_paths = {}
    for problem_name in ("moap", "mokp"):
        cfg = PROBLEM_CONFIG[problem_name]
        csv_path, out_path = resolve_paths(problem_name, cfg, args, results_dir)
        problem_data[problem_name] = load_problem_data(problem_name, cfg, csv_path)
        output_paths[problem_name] = out_path

    for problem_name in ("moap", "mokp"):
        plot_problem_grid(
            problem_name,
            PROBLEM_CONFIG[problem_name],
            problem_data[problem_name],
            args,
            output_paths[problem_name],
        )

    out_combined = (
        Path(args.out_combined).expanduser().resolve()
        if args.out_combined
        else results_dir / "combined_parallel_result_plot.png"
    )
    plot_combined_grid(problem_data, args, out_combined)


if __name__ == "__main__":
    main()
