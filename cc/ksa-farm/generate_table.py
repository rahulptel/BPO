import argparse
from pathlib import Path

BASE_COMMAND = "python $BASEPATH/src/run_ksa.py"
PROBLEM_TIME_LIMITS = {
    50: {3: 120, 4: 240, 5: 240},
    250: {3: 120, 4: 240, 5: 240},
    500: {3: 120, 4: 240, 5: 240},
}
PROBLEM_VARIANTS = tuple(
    (
        f"problem.n_items={n_items}",
        f"problem.n_objs={n_objs}",
        f"time_limit={time_limit}",
    )
    for n_items, objectives in PROBLEM_TIME_LIMITS.items()
    for n_objs, time_limit in objectives.items()
)
INSTANCES_PER_CASE = 10


class CaseSpec:
    """Encapsulates the parameters needed for a single run command."""

    __slots__ = ("case_id", "from_pid", "to_pid", "overrides")

    def __init__(self, case_id, from_pid, to_pid, overrides):
        self.case_id = case_id
        self.from_pid = from_pid
        self.to_pid = to_pid
        self.overrides = tuple(overrides)

    def render(self):
        """Return the formatted line for table.dat."""
        parts = [
            BASE_COMMAND,
            f"from_pid={self.from_pid}",
            f"to_pid={self.to_pid}",
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
    variants,
):
    """Return every CaseSpec for the provided parameter grid."""
    specs = []
    case_id = 1
    for from_pid, to_pid in pid_windows:
        for variant in variants:
            specs.append(CaseSpec(case_id, from_pid, to_pid, variant))
            case_id += 1
    return specs


def write_table(lines, output_path):
    """Write every line to the requested table.dat file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the table.dat file with KSA runs."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("table.dat"),
        help="Where to write the table.dat file (default: alongside this script).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pid_range = range(1, 26)

    pid_windows = build_pid_windows(pid_range, INSTANCES_PER_CASE)
    specs = generate_case_specs(
        pid_windows,
        PROBLEM_VARIANTS,
    )
    lines = [spec.render() for spec in specs]
    write_table(lines, args.output)


if __name__ == "__main__":
    main()
