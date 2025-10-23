from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


COMMAND_TEMPLATE = (
    "python src/run_bpo.py "
    "problem.iseed={iseed} "
    "bo.rseed={rseed} "
    "acquisition.name={acquisition} "
    "acquisition.batch_size_q=1 "
    "bo.n_iterations=50"
)


@dataclass(frozen=True)
class CaseSpec:
    """Encapsulates the parameters needed for a single run command."""

    case_id: int
    iseed: int
    rseed: int
    acquisition: str

    def render(self) -> str:
        """Return the formatted line for table.dat."""
        command = COMMAND_TEMPLATE.format(
            iseed=self.iseed,
            rseed=self.rseed,
            acquisition=self.acquisition,
        )
        return f"{self.case_id} {command}"


def generate_case_specs(
    iseed_range: Sequence[int],
    rseed_range: Sequence[int],
    acquisitions: Sequence[str],
) -> List[CaseSpec]:
    """Return every CaseSpec for the provided parameter grid."""
    specs: List[CaseSpec] = []
    case_id = 1
    for iseed in iseed_range:
        for rseed in rseed_range:
            for acquisition in acquisitions:
                specs.append(CaseSpec(case_id, iseed, rseed, acquisition))
                case_id += 1
    return specs


def write_table(lines: Iterable[str], output_path: Path) -> None:
    """Write every line to the requested table.dat file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
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


def main() -> None:
    args = parse_args()
    iseed_values = range(1, 101)
    rseed_values = range(1, 6)
    acquisitions = ("random", "qlogehvi")

    specs = generate_case_specs(iseed_values, rseed_values, acquisitions)
    lines = [spec.render() for spec in specs]
    write_table(lines, args.output)


if __name__ == "__main__":
    main()
