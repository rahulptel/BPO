import argparse
from pathlib import Path

BASE_COMMAND = "python src/run_aug_cheby.py"
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
CASE_VARIANTS = PROBLEM_VARIANTS


class CaseSpec:
    """Encapsulates the parameters needed for a single run command."""

    __slots__ = ("case_id", "iseed", "rseed", "overrides")

    def __init__(self, case_id, iseed, rseed, overrides):
        self.case_id = case_id
        self.iseed = iseed
        self.rseed = rseed
        self.overrides = tuple(overrides)

    def render(self):
        """Return the formatted line for table.dat."""
        parts = [
            BASE_COMMAND,
            f"problem.iseed={self.iseed}",
            f"rseed={self.rseed}",
        ]
        parts.extend(self.overrides)
        return f"{self.case_id} {' '.join(parts)}"


def generate_case_specs(
    iseed_range,
    rseed_range,
    variants,
):
    """Return every CaseSpec for the provided parameter grid."""
    specs = []
    case_id = 1
    for iseed in iseed_range:
        for variant in variants:
            for rseed in rseed_range:
                specs.append(CaseSpec(case_id, iseed, rseed, variant))
                case_id += 1
    return specs


def write_table(lines, output_path):
    """Write every line to the requested table.dat file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the table.dat file with Augmented Chebyshev runs."
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
    rseed_values = range(1, 6)

    specs = generate_case_specs(iseed_values, rseed_values, CASE_VARIANTS)
    lines = [spec.render() for spec in specs]
    write_table(lines, args.output)


if __name__ == "__main__":
    main()
