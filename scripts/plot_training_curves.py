from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path so we can import src.* modules when running from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.training_curves import plot_training_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot loss and mIoU curves from training logs")
    parser.add_argument("--metrics-file", type=str, default="runs/iris-segformer/metrics.csv", help="Path to metrics.csv")
    parser.add_argument("--summary-file", type=str, default="runs/iris-segformer/epoch_summary.txt", help="Fallback summary file")
    parser.add_argument("--output", type=str, default="runs/iris-segformer/visualizations/training_curves.png", help="Output plot path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = plot_training_curves(
        metrics_file=args.metrics_file,
        summary_file=args.summary_file,
        output_path=args.output,
    )
    print(f"Saved training curves to: {output_path}")


if __name__ == "__main__":
    main()