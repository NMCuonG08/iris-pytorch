from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SHOWCASE_DIRS = [
    "augmentation_visualizations",
    "batch_test_results",
    "inference_test_results",
    "overlay_examples",
    "overlay_test_results",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean showcase result folders quickly")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )
    parser.add_argument(
        "--remove-dirs",
        action="store_true",
        help="Remove entire showcase directories instead of only clearing their contents",
    )
    return parser.parse_args()


def clear_directory_contents(folder: Path) -> int:
    deleted = 0
    if not folder.exists():
        return deleted

    for child in folder.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)
        deleted += 1
    return deleted


def remove_directory(folder: Path) -> bool:
    if not folder.exists():
        return False
    shutil.rmtree(folder, ignore_errors=True)
    return True


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    target_dirs = [root / d for d in SHOWCASE_DIRS]

    print("Showcase targets:")
    for d in target_dirs:
        print(f"- {d}")

    if not args.yes:
        answer = input("Delete showcase artifacts now? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Cancelled.")
            return

    if args.remove_dirs:
        removed = 0
        for d in target_dirs:
            if remove_directory(d):
                removed += 1
        print(f"Done. Removed {removed} directories.")
    else:
        total_deleted = 0
        for d in target_dirs:
            d.mkdir(parents=True, exist_ok=True)
            total_deleted += clear_directory_contents(d)
        print(f"Done. Removed {total_deleted} artifacts, kept folder structure.")


if __name__ == "__main__":
    main()
