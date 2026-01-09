import argparse
from pathlib import Path

BASE_COMMAND = "python $BASEPATH/src/run_ea.py"
PROBLEM_TIME_LIMITS = {
    50: {3: 120, 4: 240, 5: 240},
    250: {3: 120, 4: 240, 5: 240},
    500: {3: 120, 4: 240, 5: 240},
}
INSTANCES_PER_CASE = 10


def format_seconds(seconds):
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"


MOKP_VARIANTS = tuple(
    (
        f"problem.n_items={n_items}",
        f"problem.n_objs={n_objs}",
        f"algorithm.time={format_seconds(time_limit)}",
    )
    for n_items, objectives in PROBLEM_TIME_LIMITS.items()
    for n_objs, time_limit in objectives.items()
)
MOAP_N_AGENTS = 10
MOAP_N_OBJS = 3
MOAP_TIME_LIMIT = 120
MOAP_VARIANTS = (
    (
        "problem=moap",
        f"problem.n_agents={MOAP_N_AGENTS}",
        f"problem.n_objs={MOAP_N_OBJS}",
        f"algorithm.time={format_seconds(MOAP_TIME_LIMIT)}",
    ),
)
ALGORITHM_VARIANTS = (
    ("algorithm=nsga2",),
    ("algorithm=nsga3",),
    ("algorithm=smsemoa",),
    ("algorithm=ctaea",),
)


class CaseSpec:
    """Encapsulates the parameters needed for a single run command."""

    __slots__ = ("case_id", "from_pid", "to_pid", "algorithm_seed", "overrides")

    def __init__(self, case_id, from_pid, to_pid, algorithm_seed, overrides):
        self.case_id = case_id
        self.from_pid = from_pid
        self.to_pid = to_pid
        self.algorithm_seed = algorithm_seed
        self.overrides = tuple(overrides)

    def render(self):
        """Return the formatted line for table.dat."""
        parts = [
            BASE_COMMAND,
            f"from_pid={self.from_pid}",
            f"to_pid={self.to_pid}",
            f"algorithm.seed={self.algorithm_seed}",
        ]
        parts.extend(self.overrides)
        return f"{self.case_id} {' '.join(parts)}"


def build_pid_windows(pid_range, instances_per_case):
    """Partition the PID range into non-overlapping [from_pid, to_pid) windows."""
    windows = []
    start = pid_range.start
    while start < pid_range.stop:
        stop = min(start + instances_per_case, pid_range.stop)
        windows.append((start, stop))
        start = stop
    return windows


def generate_case_specs(
    pid_windows,
    algorithm_seed_range,
    variants,
):
    """Return every CaseSpec for the provided parameter grid."""
    specs = []
    case_id = 1
    for from_pid, to_pid in pid_windows:
        for variant in variants:
            for algorithm_seed in algorithm_seed_range:
                specs.append(
                    CaseSpec(case_id, from_pid, to_pid, algorithm_seed, variant)
                )
                case_id += 1
    return specs


def write_table(lines, output_path):
    """Write every line to the requested table.dat file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def select_problem_variants(problem):
    if problem == "mokp":
        return MOKP_VARIANTS
    if problem == "moap":
        return MOAP_VARIANTS
    return MOKP_VARIANTS + MOAP_VARIANTS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the table.dat file with EA runs."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("table.dat"),
        help="Where to write the table.dat file (default: alongside this script).",
    )
    parser.add_argument(
        "--problem",
        choices=("all", "mokp", "moap"),
        default="all",
        help="Which problem cases to generate (default: all).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pid_range = range(1, 26)
    algorithm_seed_values = range(1, 6)

    pid_windows = build_pid_windows(pid_range, INSTANCES_PER_CASE)
    problem_variants = select_problem_variants(args.problem)
    case_variants = tuple(
        problem_variant + algorithm_variant
        for problem_variant in problem_variants
        for algorithm_variant in ALGORITHM_VARIANTS
    )
    specs = generate_case_specs(pid_windows, algorithm_seed_values, case_variants)
    lines = [spec.render() for spec in specs]
    write_table(lines, args.output)


if __name__ == "__main__":
    main()
