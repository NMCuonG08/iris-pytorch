#!/usr/bin/env python
"""Quick debug script to identify training issues."""

import sys
print("[TEST 1] Python interpreter:", sys.executable)

try:
    print("[TEST 2] Importing torch...")
    import torch
    print(f"  ✓ torch={torch.__version__}")
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

try:
    print("[TEST 3] Importing pathlib...")
    from pathlib import Path
    print("  ✓ pathlib imported")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

try:
    print("[TEST 4] Loading YAML config...")
    from src.utils import load_yaml_config
    config = load_yaml_config("configs/train.yaml")
    print(f"  ✓ Config loaded: data.image_ext={config['data']['image_ext']}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("[TEST 5] Checking dataset...")
    root = Path(config["data"]["root_dir"])
    img_dir = root / config["data"]["image_dir"]
    mask_dir = root / config["data"]["mask_dir"]
    img_ext = config["data"]["image_ext"]
    mask_ext = config["data"].get("mask_ext", ".png")
    
    images = list(img_dir.glob(f"*{img_ext}"))
    masks = list(mask_dir.glob(f"*{mask_ext}"))
    
    print(f"  ✓ Images found: {len(images)}")
    print(f"  ✓ Masks found: {len(masks)}")
    
    if len(images) == 0:
        print("  ✗ No images found!")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("[TEST 6] Creating dataloaders (no workers)...")
    from src.data import create_dataloaders
    train_loader, val_loader = create_dataloaders(config)
    print(f"  ✓ Train loader: {len(train_loader)} batches")
    print(f"  ✓ Val loader: {len(val_loader)} batches")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("[TEST 7] Loading first batch...")
    for batch_idx, (images_batch, masks_batch) in enumerate(train_loader):
        print(f"  ✓ Batch {batch_idx}: images={images_batch.shape}, masks={masks_batch.shape}")
        if batch_idx >= 0:  # Just test first batch
            break
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed! Issue is likely in model/trainer initialization")
