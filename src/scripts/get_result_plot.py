#!/usr/bin/env python3
"""Aggregate BPO, KSA, and AugCheby hypervolume traces across seeds and produce summary plots."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize BPO, KSA, and AugCheby results by averaging hypervolume over seeds."
    )
    parser.add_argument(
        "--n-objs",
        nargs="+",
        type=int,
        default=[3, 4, 5],
        help="Objective counts to include (default: 3 4 5).",
    )
    parser.add_argument(
        "--items",
        nargs="+",
        type=int,
        default=[50, 250, 500],
        help="Item sizes to include (default: 50 250 500).",
    )
    parser.add_argument(
        "--plots-per-figure",
        type=int,
        default=9,
        help="Maximum number of panels per figure (default: 9 for a 3x3 grid).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=25,
        help="Maximum number of iterations to plot per run (default: 25).",
    )
    parser.add_argument(
        "--output-prefix",
        default="bpo_hv_summary",
        help="Prefix for saved figure filenames (default: bpo_hv_summary).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures after saving instead of closing them.",
    )
    return parser.parse_args()


def collect_run_paths(root, items, n_objs, file_glob):
    if not root.exists():
        return []
    pattern = f"mokp-items-{items}_objs-{n_objs}_iseed-*"
    run_paths = []
    for iseed_dir in sorted(root.glob(pattern)):
        iseed = parse_iseed(iseed_dir)
        if iseed is not None and iseed > 10:
            continue
        run_paths.extend(sorted(iseed_dir.rglob(file_glob)))
    return run_paths


def parse_iseed(path):
    for part in path.parts:
        if part.startswith("mokp-items-") and "_objs-" in part and "_iseed-" in part:
            try:
                return int(part.split("_iseed-")[-1])
            except ValueError:
                return None
    return None


def parse_run_hyperparams(path):
    params = {}
    for part in path.parts:
        if part.startswith("batch_size_q-"):
            params["batch_size_q"] = part.split("batch_size_q-")[-1]
        elif part.startswith("sequential-"):
            params["sequential"] = part.split("sequential-")[-1]
    return params


def filter_run_paths(run_paths, required_params):
    if not required_params:
        return run_paths
    filtered = []
    for path in run_paths:
        params = parse_run_hyperparams(path)
        if all(params.get(key) == str(val) for key, val in required_params.items()):
            filtered.append(path)
    return filtered


def load_run_traces(run_paths, max_iterations=None):
    traces = []
    for path in run_paths:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        hv_values = []
        iterations = []
        records = payload.get("iterations")
        if records is None:
            records = payload.get("iter_records", [])
        for record in records:
            iteration = record.get("n_evaluation")
            hv = record.get("hv")
            if iteration is None or hv is None:
                continue
            iterations.append(int(iteration))
            hv_values.append(float(hv))
            if max_iterations is not None and len(iterations) >= max_iterations:
                break

        if hv_values:
            traces.append({"iterations": iterations, "hv": hv_values})
    return traces


def aggregate_trace(traces):
    if not traces:
        return [], []
    max_len = max(len(trace["hv"]) for trace in traces)
    if max_len == 0:
        return [], []
    padded = []
    for trace in traces:
        hv_values = trace["hv"]
        if not hv_values:
            continue
        if len(hv_values) < max_len:
            hv_values = hv_values + [hv_values[-1]] * (max_len - len(hv_values))
        padded.append(hv_values)
    if not padded:
        return [], []
    avg_hv = np.mean(np.array(padded), axis=0).tolist()
    iterations = None
    for trace in traces:
        if len(trace["iterations"]) == max_len:
            iterations = trace["iterations"]
            break
    if iterations is None:
        iterations = list(range(1, max_len + 1))
    return iterations, [float(val) for val in avg_hv]


def iter_chunks(seq, size):
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]


def plot_traces(
    algo_cfgs,
    combinations,
    plots_per_figure,
    output_prefix,
    show,
    max_iterations,
    output_dir,
):
    saved = []
    for figure_idx, chunk in enumerate(iter_chunks(combinations, plots_per_figure), 1):
        n_panels = len(chunk)
        if n_panels == 0:
            continue
        n_cols = min(3, n_panels)
        n_rows = math.ceil(n_panels / n_cols)
        figsize = (5 * n_cols, 3.6 * n_rows)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes_flat = axes.flatten()

        for (n_objs, items), ax in zip(chunk, axes_flat):
            ax.set_title(f"objs={n_objs}, items={items}", fontsize=11)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Hypervolume")
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
            handles = []

            for cfg in algo_cfgs:
                run_paths = collect_run_paths(
                    cfg["root"], items, n_objs, cfg["file_glob"]
                )
                run_paths = filter_run_paths(run_paths, cfg.get("filters", {}))
                traces = load_run_traces(run_paths, max_iterations=max_iterations)
                iterations, hv_mean = aggregate_trace(traces)
                runs = len(run_paths)

                if iterations:
                    alpha = cfg.get("alpha", 0.7)
                    if cfg.get("marker"):
                        ax.scatter(
                            iterations,
                            hv_mean,
                            color=cfg["color"],
                            s=26,
                            alpha=alpha,
                            marker=cfg["marker"],
                        )
                    (line,) = ax.plot(
                        iterations,
                        hv_mean,
                        color=cfg["color"],
                        linewidth=1.6,
                        alpha=alpha,
                        label=f"{cfg['label']} (runs={runs})",
                    )
                    handles.append(line)
                    print(
                        f"[{cfg['label']}] objs={n_objs}, items={items} | "
                        f"runs={runs} iterations={len(iterations)} (max={iterations[-1]})"
                    )
                else:
                    print(
                        f"[{cfg['label']}] objs={n_objs}, items={items} | no runs found."
                    )

            if handles:
                ax.legend(loc="lower right")
            else:
                ax.set_axis_off()

        for ax in axes_flat[n_panels:]:
            ax.set_axis_off()

        fig.tight_layout()
        output_path = output_dir / f"{output_prefix}_part{figure_idx}.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        saved.append(output_path)
        print(f"Saved figure to {output_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    return saved


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    outputs_dir = script_dir.parents[1] / "outputs"
    results_dir = script_dir.parents[1] / "results"
    bpo_root = outputs_dir / "bpo"
    aug_root = outputs_dir / "aug_cheby"
    ksa_root = outputs_dir / "ksa"
    if (ksa_root / "ksa").exists():
        ksa_root = ksa_root / "ksa"

    if not bpo_root.exists():
        raise FileNotFoundError(f"Expected BPO results under {bpo_root}")
    if not aug_root.exists():
        print(f"Warning: AugCheby results not found under {aug_root}")
    if not ksa_root.exists():
        print(f"Warning: KSA results not found under {ksa_root}")

    algo_cfgs = [
        {
            "label": "BPO q=1",
            "root": bpo_root,
            "file_glob": "run_bo_*.json",
            "color": "#1f77b4",
            "filters": {"batch_size_q": "1"},
            "marker": "o",
        },
        {
            "label": "AugCheby",
            "root": aug_root,
            "file_glob": "run_aug_cheby_*.json",
            "color": "#ff7f0e",
            "filters": {},
            "marker": "s",
        },
        {
            "label": "KSA",
            "root": ksa_root,
            "file_glob": "run_ksa_*.json",
            "color": "#9467bd",
            "filters": {},
            "marker": "^",
        },
    ]

    # {
    #     "label": "BPO q=2 seq",
    #     "root": bpo_root,
    #     "file_glob": "run_bo_*.json",
    #     "color": "#2ca02c",
    #     "filters": {"batch_size_q": "2", "sequential": "True"},
    #     "marker": "o",
    # },
    # {
    #     "label": "BPO q=2 joint",
    #     "root": bpo_root,
    #     "file_glob": "run_bo_*.json",
    #     "color": "#d62728",
    #     "filters": {"batch_size_q": "2", "sequential": "False"},
    #     "marker": "o",
    # },

    combinations = [
        (n_objs, items)
        for items in sorted(set(args.items))
        for n_objs in sorted(set(args.n_objs))
    ]
    if not combinations:
        raise ValueError(
            "No combinations to process. Provide --items/--n-objs with valid values."
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = plot_traces(
        algo_cfgs,
        combinations,
        args.plots_per_figure,
        args.output_prefix,
        args.show,
        args.max_iterations,
        results_dir,
    )

    if not saved_paths:
        print("No figures were generated. Check that the requested results exist.")


if __name__ == "__main__":
    main()
