"""Generate writeup sections from experiment logs.

This script reads results/experiments.csv and generates formatted
sections for the writeup document.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logging import configure_logging, get_logger

logger = get_logger("scripts.generate_writeup_sections")


def extract_best_scores(experiments_file: Path) -> dict:
    """Extract best CV scores per task from experiments log.

    Args:
        experiments_file: Path to experiments.csv.

    Returns:
        Dictionary mapping task to best score and hyperparameters.
    """
    if not experiments_file.exists():
        logger.warning(f"Experiments file not found: {experiments_file}")
        return {}

    df = pd.read_csv(experiments_file)

    best_experiments = {}
    for task in ["read", "category", "rating"]:
        task_df = df[df["task"] == task]
        if len(task_df) == 0:
            continue

        best_idx = task_df["best_score"].idxmax()
        best_row = task_df.loc[best_idx]

        best_experiments[task] = {
            "best_score": best_row["best_score"],
            "model_type": best_row["model_type"],
            "hyperparams": json.loads(best_row["hyperparams_json"]),
            "cv_scores": json.loads(best_row["cv_scores_json"]),
            "seed": best_row["seed"],
        }

    return best_experiments


def generate_results_section(best_experiments: dict) -> str:
    """Generate results section markdown.

    Args:
        best_experiments: Dictionary of best experiments per task.

    Returns:
        Markdown formatted results section.
    """
    lines = ["## 5. Results", "", "### Cross-Validation Scores", ""]

    for task in ["read", "category", "rating"]:
        if task not in best_experiments:
            continue

        exp = best_experiments[task]
        lines.append(f"**{task.capitalize()} Task:**")
        lines.append(f"- Model: {exp['model_type']}")
        lines.append(f"- Best CV Score: {exp['best_score']:.4f}")
        lines.append(f"- CV Fold Scores: {exp['cv_scores']}")
        lines.append(f"- Hyperparameters: {exp['hyperparams']}")
        lines.append("")

    return "\n".join(lines)


def update_writeup(
    writeup_file: Path, experiments_file: Path, output_file: Path
) -> None:
    """Update writeup with results from experiments.

    Args:
        writeup_file: Path to writeup template.
        experiments_file: Path to experiments.csv.
        output_file: Path to output writeup.
    """
    logger.info("Generating writeup sections from experiments")

    # Read existing writeup
    if writeup_file.exists():
        with open(writeup_file, "r", encoding="utf-8") as f:
            writeup_content = f.read()
    else:
        logger.warning(f"Writeup file not found: {writeup_file}")
        writeup_content = ""

    # Extract best scores
    best_experiments = extract_best_scores(experiments_file)

    # Generate results section
    results_section = generate_results_section(best_experiments)

    # Replace or append results section
    if "## 5. Results" in writeup_content:
        # Replace existing section
        lines = writeup_content.split("\n")
        start_idx = None
        end_idx = None
        for i, line in enumerate(lines):
            if line.startswith("## 5. Results"):
                start_idx = i
            elif start_idx is not None and line.startswith("## ") and i > start_idx:
                end_idx = i
                break

        if start_idx is not None:
            if end_idx is not None:
                lines = (
                    lines[:start_idx]
                    + results_section.split("\n")
                    + lines[end_idx:]
                )
            else:
                lines = lines[:start_idx] + results_section.split("\n")
            writeup_content = "\n".join(lines)
    else:
        # Append results section
        writeup_content += "\n\n" + results_section

    # Write updated writeup
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(writeup_content)

    logger.info(f"Updated writeup saved to {output_file}")


def main() -> int:
    """Main entry point for writeup generation.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Generate writeup sections from experiments"
    )
    parser.add_argument(
        "--writeup",
        type=str,
        default="writeup.txt",
        help="Path to writeup template",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default="results/experiments.csv",
        help="Path to experiments.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="writeup.txt",
        help="Path to output writeup",
    )

    args = parser.parse_args()

    configure_logging()

    update_writeup(
        Path(args.writeup), Path(args.experiments), Path(args.output)
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())


