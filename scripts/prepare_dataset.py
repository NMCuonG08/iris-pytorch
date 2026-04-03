import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_files(root: Path, allow_parent_fallback: bool = False) -> list[Path]:
    if root.exists() and root.is_dir():
        candidates = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
        if candidates:
            return sorted(candidates)

    if allow_parent_fallback:
        parent = root.parent
    else:
        parent = None

    if parent is not None and parent.exists():
        candidates = [path for path in parent.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
        if candidates:
            return sorted(candidates)

    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare UBIRIS-like dataset into project structure")
    parser.add_argument("--source-images", type=str, required=True, help="Path to raw images folder")
    parser.add_argument("--source-masks", type=str, default="", help="Path to raw masks folder (optional for image-only archives)")
    parser.add_argument("--target-root", type=str, default="dataset", help="Target dataset root")
    parser.add_argument("--image-ext", type=str, default=".jpg", help="Image extension in output")
    parser.add_argument("--mask-ext", type=str, default=".png", help="Mask extension in output")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of move")
    return parser.parse_args()


def write_split(file_path: Path, stems: list[str]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        for stem in stems:
            handle.write(stem + "\n")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    src_images = Path(args.source_images)
    target_root = Path(args.target_root)
    out_images = target_root / "images"
    out_masks = target_root / "masks"

    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    image_paths = collect_files(src_images, allow_parent_fallback=True)
    if not image_paths:
        raise ValueError(
            f"No images found under {src_images}.\n"
            f"Check the extracted UBIRIS folder path and make sure it contains image files."
        )

    mask_index: dict[str, Path] = {}
    masks_enabled = bool(args.source_masks)
    if masks_enabled:
        src_masks = Path(args.source_masks)
        mask_paths = collect_files(src_masks, allow_parent_fallback=False)
        mask_index = {path.stem: path for path in mask_paths}
        if not mask_index:
            raise ValueError(
                f"No masks found under {src_masks}.\n"
                f"The archive appears to be image-only. If you only want to split images, omit --source-masks."
            )

    image_index = {path.stem: path for path in image_paths}

    valid_stems: list[str] = []
    for image_path in image_paths:
        stem = image_path.stem
        # Try to match mask with or without "OperatorA_" prefix
        mask_stem = stem
        if not masks_enabled:
            valid_stems.append(stem)
        elif mask_index.get(stem) is not None:
            valid_stems.append(stem)
        elif mask_index.get(f"OperatorA_{stem}") is not None:
            # Found mask with OperatorA_ prefix, store both stems for later retrieval
            mask_stem = f"OperatorA_{stem}"
            valid_stems.append(stem)
            # Update mask_index to use image stem as key for consistent lookup
            if stem not in mask_index:
                mask_index[stem] = mask_index[mask_stem]

    if not valid_stems:
        raise ValueError("No matching image/mask pairs found")

    for stem in valid_stems:
        src_img = image_index[stem]
        dst_img = out_images / src_img.name

        if args.copy:
            shutil.copy2(src_img, dst_img)
            if masks_enabled:
                src_msk = mask_index[stem]
                dst_msk = out_masks / src_msk.name
                shutil.copy2(src_msk, dst_msk)
        else:
            shutil.move(str(src_img), str(dst_img))
            if masks_enabled:
                src_msk = mask_index[stem]
                dst_msk = out_masks / src_msk.name
                shutil.move(str(src_msk), str(dst_msk))

    random.shuffle(valid_stems)
    total = len(valid_stems)
    train_end = int(total * args.train_ratio)
    val_end = train_end + int(total * args.val_ratio)

    train_stems = valid_stems[:train_end]
    val_stems = valid_stems[train_end:val_end]
    test_stems = valid_stems[val_end:]

    splits_dir = target_root / "splits"
    write_split(splits_dir / "train.txt", train_stems)
    write_split(splits_dir / "val.txt", val_stems)
    write_split(splits_dir / "test.txt", test_stems)

    print(f"Prepared {total} pairs")
    print(f"Train/Val/Test: {len(train_stems)}/{len(val_stems)}/{len(test_stems)}")
    print(f"Output images: {out_images}")
    if masks_enabled:
        print(f"Output masks: {out_masks}")
    else:
        print("Masks were not provided; only image splits were generated.")


if __name__ == "__main__":
    main()
