#!/usr/bin/env python3
"""Run RGB NAFNet defogging inference with the released checkpoints.

This script is intentionally self-contained for the Kaggle model-weights
dataset. Keep `nafnet_arch.py` in the same directory as this file.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nafnet_arch import NAFNet


VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class ResidualNAFNetRGB(torch.nn.Module):
    """RGB restoration wrapper used by the released checkpoints."""

    def __init__(self, width: int, middle_blocks: int, enc_blocks: list[int], dec_blocks: list[int]) -> None:
        super().__init__()
        self.core = NAFNet(
            in_channels=3,
            out_channels=3,
            width=width,
            middle_blk_num=middle_blocks,
            enc_blk_nums=enc_blocks,
            dec_blk_nums=dec_blocks,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.core(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run defogging inference with a released NAFNet checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth/.pt checkpoint file.")
    parser.add_argument("--input", required=True, help="Input image file or directory of images.")
    parser.add_argument("--output-dir", required=True, help="Directory for restored outputs.")
    parser.add_argument("--model-config", default=None, help="Optional JSON config. If omitted, checkpoint provenance is used.")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size for large images.")
    parser.add_argument("--tile-overlap", type=int, default=64, help="Overlap used for tiled blending.")
    parser.add_argument("--save-comparison", action="store_true", help="Also save input/restored side-by-side previews.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"{path} is expected to contain a dict with key 'model_state_dict'")
    return checkpoint


def load_model_config(config_path: Path | None, checkpoint: dict) -> dict:
    if config_path is not None:
        return json.loads(config_path.read_text(encoding="utf-8"))

    provenance = checkpoint.get("provenance", {})
    if isinstance(provenance, dict):
        if isinstance(provenance.get("config"), dict):
            return provenance["config"]
        if {"width", "middle_blocks", "enc_blocks", "dec_blocks"}.issubset(provenance):
            return provenance

    return {
        "model_type": "residual_rgb",
        "width": 32,
        "middle_blocks": 12,
        "enc_blocks": [2, 2, 4, 8],
        "dec_blocks": [2, 2, 2, 2],
    }


def build_model(config: dict) -> torch.nn.Module:
    width = int(config["width"])
    middle_blocks = int(config["middle_blocks"])
    enc_blocks = [int(x) for x in config["enc_blocks"]]
    dec_blocks = [int(x) for x in config["dec_blocks"]]
    model_type = str(config.get("model_type", "residual_rgb"))
    if model_type not in {"residual_rgb", "ResidualNAFNetRGB"}:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return ResidualNAFNetRGB(width, middle_blocks, enc_blocks, dec_blocks)


def collect_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in VALID_SUFFIXES)
    raise FileNotFoundError(path)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        Image.fromarray(array).save(path, quality=95)
    else:
        Image.fromarray(array).save(path)


def iter_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, tile_size - overlap)
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def tiled_inference(model: torch.nn.Module, rgb: np.ndarray, tile_size: int, tile_overlap: int, device: torch.device) -> np.ndarray:
    height, width = rgb.shape[:2]
    output = np.zeros((height, width, 3), dtype=np.float32)
    weight = np.zeros((height, width, 1), dtype=np.float32)
    y_starts = iter_starts(height, tile_size, tile_overlap)
    x_starts = iter_starts(width, tile_size, tile_overlap)

    with torch.no_grad():
        for top in y_starts:
            for left in x_starts:
                patch = rgb[top : top + tile_size, left : left + tile_size, :]
                patch_tensor = torch.from_numpy(np.moveaxis(patch, -1, 0)).unsqueeze(0).float().to(device)
                pred = model(patch_tensor).squeeze(0).detach().cpu().numpy()
                pred = np.moveaxis(np.clip(pred, 0.0, 1.0), 0, -1)

                patch_h, patch_w = patch.shape[:2]
                blend = np.ones((patch_h, patch_w, 1), dtype=np.float32)
                ramp = min(tile_overlap, patch_h // 2, patch_w // 2)
                if ramp > 0:
                    y = np.minimum(np.arange(patch_h), np.arange(patch_h)[::-1]).astype(np.float32)
                    x = np.minimum(np.arange(patch_w), np.arange(patch_w)[::-1]).astype(np.float32)
                    y = np.clip(y / max(1, ramp), 0.0, 1.0)
                    x = np.clip(x / max(1, ramp), 0.0, 1.0)
                    blend = np.minimum.outer(y, x)[:, :, None]
                    blend = np.clip(blend, 1e-3, 1.0)

                output[top : top + patch_h, left : left + patch_w, :] += pred * blend
                weight[top : top + patch_h, left : left + patch_w, :] += blend

    return output / np.clip(weight, 1e-6, None)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = load_model_config(Path(args.model_config) if args.model_config else None, checkpoint)

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = collect_inputs(Path(args.input))
    if not input_paths:
        raise FileNotFoundError(f"No images found in {args.input}")

    rows = []
    start = time.time()
    for input_path in input_paths:
        output_path = output_dir / f"{input_path.stem}_defogged.png"
        compare_path = output_dir / f"{input_path.stem}_comparison.jpg"
        if output_path.exists() and not args.overwrite:
            rows.append({"input": str(input_path), "output": str(output_path), "status": "skipped_existing"})
            continue

        rgb = load_rgb(input_path)
        restored = tiled_inference(model, rgb, args.tile_size, args.tile_overlap, device)
        save_rgb(output_path, restored)
        if args.save_comparison:
            save_rgb(compare_path, np.concatenate([rgb, restored], axis=1))
        rows.append({"input": str(input_path), "output": str(output_path), "status": "written"})

    manifest = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "model_config": {
            "model_type": config.get("model_type", "residual_rgb"),
            "width": int(config["width"]),
            "middle_blocks": int(config["middle_blocks"]),
            "enc_blocks": [int(x) for x in config["enc_blocks"]],
            "dec_blocks": [int(x) for x in config["dec_blocks"]],
        },
        "elapsed_seconds": time.time() - start,
        "rows": rows,
    }
    (output_dir / "inference_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
