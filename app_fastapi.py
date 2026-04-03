from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from src.utils.config import load_yaml_config
from src.utils.visualization import load_segformer_checkpoint, predict_mask


def _encode_png_base64(image_rgb: np.ndarray) -> str:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode image to PNG")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def _overlay_prediction(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = image_rgb.astype(np.float32).copy()
    color = np.array([0, 255, 0], dtype=np.float32)
    m = mask.astype(bool)
    overlay[m] = (1.0 - alpha) * overlay[m] + alpha * color
    return np.clip(overlay, 0, 255).astype(np.uint8)


app = FastAPI(title="Iris Segmentation API", version="1.0.0")

_CONFIG_PATH = os.getenv("CONFIG_PATH", "configs/train.yaml")
_CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "runs/iris-segformer/checkpoints/best.pt")
_DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

config = load_yaml_config(_CONFIG_PATH)
data_cfg = config["data"]
model_cfg = config["model"]
input_size = tuple(data_cfg.get("input_size", [512, 512]))

if not Path(_CHECKPOINT_PATH).exists():
    raise FileNotFoundError(f"Checkpoint not found: {_CHECKPOINT_PATH}")

model = load_segformer_checkpoint(
    checkpoint_path=_CHECKPOINT_PATH,
    backbone_name=model_cfg["backbone_name"],
    num_classes=model_cfg.get("num_classes", 1),
    decoder_channels=model_cfg.get("decoder_channels", 256),
    dropout=model_cfg.get("dropout", 0.1),
    device=_DEVICE,
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": _DEVICE,
        "config": _CONFIG_PATH,
        "checkpoint": _CHECKPOINT_PATH,
        "input_size": list(input_size),
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!doctype html>
<html>
  <head>
    <meta charset='utf-8' />
    <meta name='viewport' content='width=device-width, initial-scale=1' />
    <title>Iris Segmentation UI</title>
    <style>
      body { font-family: Segoe UI, Arial, sans-serif; max-width: 960px; margin: 24px auto; padding: 0 12px; }
      h1 { margin-bottom: 8px; }
      .row { display: flex; gap: 16px; flex-wrap: wrap; }
      .card { border: 1px solid #ddd; border-radius: 8px; padding: 12px; flex: 1; min-width: 280px; }
      img { width: 100%; border-radius: 6px; border: 1px solid #eee; }
      button { padding: 10px 16px; border: 0; border-radius: 6px; background: #0a66c2; color: white; cursor: pointer; }
      button:disabled { background: #8aaed1; cursor: not-allowed; }
      .muted { color: #666; font-size: 14px; }
    </style>
  </head>
  <body>
    <h1>Iris Segmentation</h1>
    <p class='muted'>Upload image to predict iris mask from best.pt</p>

    <form id='predictForm'>
      <input type='file' id='fileInput' name='file' accept='image/*' required />
      <input type='number' id='threshold' name='threshold' step='0.01' min='0' max='1' value='0.5' />
      <button id='submitBtn' type='submit'>Predict</button>
    </form>

    <p id='status' class='muted'></p>

    <div class='row'>
      <div class='card'>
        <h3>Mask</h3>
        <img id='maskImg' alt='mask result' />
      </div>
      <div class='card'>
        <h3>Overlay</h3>
        <img id='overlayImg' alt='overlay result' />
      </div>
    </div>

    <script>
      const form = document.getElementById('predictForm');
      const statusEl = document.getElementById('status');
      const submitBtn = document.getElementById('submitBtn');
      const maskImg = document.getElementById('maskImg');
      const overlayImg = document.getElementById('overlayImg');

      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const fileInput = document.getElementById('fileInput');
        const threshold = document.getElementById('threshold').value;
        if (!fileInput.files.length) return;

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('threshold', threshold);

        submitBtn.disabled = true;
        statusEl.textContent = 'Predicting...';

        try {
          const res = await fetch('/predict', { method: 'POST', body: formData });
          const data = await res.json();
          if (!res.ok) throw new Error(data.detail || 'Prediction failed');

          maskImg.src = 'data:image/png;base64,' + data.mask_base64;
          overlayImg.src = 'data:image/png;base64,' + data.overlay_base64;
          statusEl.textContent = `Done. shape=${data.shape[1]}x${data.shape[0]}`;
        } catch (err) {
          statusEl.textContent = 'Error: ' + err.message;
        } finally {
          submitBtn.disabled = false;
        }
      });
    </script>
  </body>
</html>
"""


@app.post("/predict")
async def predict(file: UploadFile = File(...), threshold: float = Form(0.5)) -> JSONResponse:
    raw = await file.read()
    array = np.frombuffer(raw, dtype=np.uint8)
    image_bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return JSONResponse(status_code=400, content={"detail": "Invalid image file"})

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    pred_small = predict_mask(
        model=model,
        image_rgb=image_rgb,
        input_size=input_size,
        device=_DEVICE,
        threshold=threshold,
    )

    pred_mask = cv2.resize(pred_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    overlay = _overlay_prediction(image_rgb, pred_mask)
    mask_rgb = np.stack([pred_mask * 255, pred_mask * 255, pred_mask * 255], axis=-1)

    return JSONResponse(
        {
            "mask_base64": _encode_png_base64(mask_rgb),
            "overlay_base64": _encode_png_base64(overlay),
            "shape": [h, w],
            "threshold": threshold,
        }
    )