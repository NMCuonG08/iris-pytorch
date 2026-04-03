import argparse
import os
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract UBIRIS V2 dataset archive")
    parser.add_argument(
        "--url",
        type=str,
        default="",
        help="Direct URL to UBIRIS V2 zip archive (or any equivalent mirror archive)",
    )
    parser.add_argument(
        "--zip-path",
        type=str,
        default="",
        help="Existing local zip path. If provided, download step is skipped.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="dataset/raw",
        help="Output directory for extracted dataset",
    )
    parser.add_argument(
        "--kaggle-dataset",
        type=str,
        default="",
        help="Kaggle dataset slug, e.g. chinmoyslg/ubirisv2",
    )
    parser.add_argument(
        "--zip-password",
        type=str,
        default="",
        help="Password for encrypted zip file (if required)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    archive_path: Path | None = None
    
    if args.kaggle_dataset:
        import kagglehub
        
        kaggle_user = os.getenv("KAGGLE_USERNAME")
        kaggle_key = os.getenv("KAGGLE_KEY")
        if not kaggle_user or not kaggle_key:
            raise ValueError(
                "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY in .env file."
            )
        
        print(f"Downloading Kaggle dataset: {args.kaggle_dataset}")
        dataset_path = kagglehub.dataset_download(args.kaggle_dataset)
        print(f"Downloaded to: {dataset_path}")
        
        # Find the largest zip file in downloaded path
        zip_files = sorted(Path(dataset_path).glob("*.zip"), key=lambda p: p.stat().st_size, reverse=True)
        if zip_files:
            archive_path = zip_files[0]
            shutil.copy2(archive_path, out_dir / archive_path.name)
            archive_path = out_dir / archive_path.name
        else:
            # If no zip, check if files are already extracted
            print("No zip archive found. Dataset might already be extracted.")
            images = list(Path(dataset_path).glob("**/images/*"))
            masks = list(Path(dataset_path).glob("**/masks/*"))
            if images and masks:
                print(f"Found {len(images)} images and {len(masks)} masks")
                print("Copying to dataset folder...")
                shutil.copytree(
                    Path(dataset_path) / "images",
                    out_dir.parent / "images",
                    dirs_exist_ok=True
                )
                shutil.copytree(
                    Path(dataset_path) / "masks",
                    out_dir.parent / "masks",
                    dirs_exist_ok=True
                )
                print("Done.")
                return
            
    if args.zip_path:
        archive_path = Path(args.zip_path)
        if not archive_path.exists():
            raise FileNotFoundError(f"Zip file not found: {archive_path}")
    elif args.url:
        tmp_zip = out_dir / "dataset_download.zip"
        print(f"Downloading from: {args.url}")
        urlretrieve(args.url, str(tmp_zip))
        archive_path = tmp_zip
    
    if not archive_path:
        if not args.kaggle_dataset:
            raise ValueError(
                "No data source provided. Use --url, --zip-path, or --kaggle-dataset."
            )
        return

    print(f"Extracting {archive_path} to {out_dir}")
    
    # Handle password-protected ZIP files
    zip_password = args.zip_password.encode() if args.zip_password else None
    
    try:
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(out_dir, pwd=zip_password)
    except RuntimeError as e:
        if "encrypted" in str(e).lower() and not zip_password:
            print("⚠️ ZIP file is encrypted. Please provide password:")
            password_input = input("Enter ZIP password: ").strip()
            if password_input:
                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    zip_ref.extractall(out_dir, pwd=password_input.encode())
            else:
                raise ValueError("Password required but not provided.")
        else:
            raise

    if archive_path.name == "dataset_download.zip" and archive_path.exists():
        archive_path.unlink()

    print("Done. Raw dataset is ready at:", out_dir)


if __name__ == "__main__":
    main()
