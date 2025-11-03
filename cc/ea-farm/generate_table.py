import argparse
from pathlib import Path

BASE_COMMAND = "python src/run_ea.py"
PROBLEM_TIME_LIMITS = {
    50: {3: 120, 4: 240, 5: 240},
    250: {3: 120, 4: 240, 5: 240},
    500: {3: 120, 4: 240, 5: 240},
}


def format_seconds(seconds):
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"


PROBLEM_VARIANTS = tuple(
    (
        f"problem.n_items={n_items}",
        f"problem.n_objs={n_objs}",
        f"algorithm.time={format_seconds(time_limit)}",
    )
    for n_items, objectives in PROBLEM_TIME_LIMITS.items()
    for n_objs, time_limit in objectives.items()
)
ALGORITHM_VARIANTS = (
    ("algorithm=nsga2",),
    ("algorithm=nsga3",),
    ("algorithm=smsemoa",),
    ("algorithm=ctaea",),
)
CASE_VARIANTS = tuple(
    problem_variant + algorithm_variant
    for problem_variant in PROBLEM_VARIANTS
    for algorithm_variant in ALGORITHM_VARIANTS
)


class CaseSpec:
    """Encapsulates the parameters needed for a single run command."""

    __slots__ = ("case_id", "iseed", "algorithm_seed", "overrides")

    def __init__(self, case_id, iseed, algorithm_seed, overrides):
        self.case_id = case_id
        self.iseed = iseed
        self.algorithm_seed = algorithm_seed
        self.overrides = tuple(overrides)

    def render(self):
        """Return the formatted line for table.dat."""
        parts = [
            BASE_COMMAND,
            f"problem.iseed={self.iseed}",
            f"algorithm.seed={self.algorithm_seed}",
        ]
        parts.extend(self.overrides)
        return f"{self.case_id} {' '.join(parts)}"


def generate_case_specs(
    iseed_range,
    algorithm_seed_range,
    variants,
):
    """Return every CaseSpec for the provided parameter grid."""
    specs = []
    case_id = 1
    for iseed in iseed_range:
        for variant in variants:
            for algorithm_seed in algorithm_seed_range:
                specs.append(CaseSpec(case_id, iseed, algorithm_seed, variant))
                case_id += 1
    return specs


def write_table(lines, output_path):
    """Write every line to the requested table.dat file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


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
    return parser.parse_args()


def main():
    args = parse_args()
    iseed_values = range(1, 26)
    algorithm_seed_values = range(1, 6)

    specs = generate_case_specs(iseed_values, algorithm_seed_values, CASE_VARIANTS)
    lines = [spec.render() for spec in specs]
    write_table(lines, args.output)


if __name__ == "__main__":
    main()
