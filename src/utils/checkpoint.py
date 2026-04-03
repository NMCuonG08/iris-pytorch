from pathlib import Path

import torch


def save_checkpoint(state: dict, path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_path)
