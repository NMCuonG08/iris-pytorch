"""Shortcut to download UBIRIS V2 dataset from Kaggle using kagglehub."""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from download_dataset import main

if __name__ == "__main__":
    import sys
    
    # Auto-inject UBIRIS V2 dataset slug and password
    sys.argv = [
        sys.argv[0],
        "--kaggle-dataset",
        "chinmoyslg/ubirisv2",
        "--zip-password",
        "UBIRIS2_IEEETPAMI_101109_200966",
    ]
    main()
