import argparse
from pathlib import Path

COMMAND_TEMPLATE = (
    "python src/run_bpo.py "
    "problem.iseed={iseed} "
    "bo.rseed={rseed} "
    "problem.n_objs={n_objs} "
    "acquisition.name=random "
    "acquisition.batch_size_q=1 "
    "bo.n_iterations=50"
)


class CaseSpec:
    """Encapsulates the parameters needed for a single run command."""

    __slots__ = ("case_id", "iseed", "rseed", "n_objs")

    def __init__(self, case_id, iseed, rseed, n_objs):
        self.case_id = case_id
        self.iseed = iseed
        self.rseed = rseed
        self.n_objs = n_objs

    def render(self):
        """Return the formatted line for table.dat."""
        command = COMMAND_TEMPLATE.format(
            iseed=self.iseed,
            rseed=self.rseed,
            n_objs=self.n_objs,
        )
        return f"{self.case_id} {command}"


def generate_case_specs(
    iseed_range,
    rseed_range,
    n_objs_range,
):
    """Return every CaseSpec for the provided parameter grid."""
    specs = []
    case_id = 1
    for iseed in iseed_range:
        for rseed in rseed_range:
            for n_obj in n_objs_range:
                specs.append(CaseSpec(case_id, iseed, rseed, n_obj))
                case_id += 1
    return specs


def write_table(lines, output_path):
    """Write every line to the requested table.dat file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the table.dat file with configured BO runs."
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
    iseed_values = range(1, 101)
    rseed_values = range(1, 6)
    n_objs_values = range(2, 6)

    specs = generate_case_specs(iseed_values, rseed_values, n_objs_values)
    lines = [spec.render() for spec in specs]
    write_table(lines, args.output)


if __name__ == "__main__":
    main()
