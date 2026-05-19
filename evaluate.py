import argparse
from textwrap import indent

from src.reporting import load_result_summaries, render_markdown_results_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize archived experiment metrics for the public research repository."
    )
    parser.add_argument(
        "--format",
        choices=("text", "markdown"),
        default="text",
        help="Output format for the summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = load_result_summaries()

    if args.format == "markdown":
        print(render_markdown_results_table())
        return

    lines = ["Archived experiment summary:"]
    for experiment_name, metrics in summaries.items():
        formatted_name = experiment_name.replace("_", " ").title()
        lines.append(f"- {formatted_name}")
        for key, value in metrics.items():
            if value is None:
                lines.append(f"  - {key}: not archived")
            else:
                lines.append(f"  - {key}: {100.0 * value:.2f}%")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
