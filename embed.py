"""
Step 1 of 3: turn images into frozen I-JEPA embeddings.

Downloads Meta's pretrained I-JEPA encoder (~2.5 GB, first run only), runs every
image through it once, and saves the result to disk.

The encoder is FROZEN. We never train it -- it only describes images. Training
happens later in train.py, on top of these saved embeddings.

Output shape is (N, 256, 1280): for each image, a 16x16 grid of patches, each
described by 1280 numbers. We keep it at patch level rather than averaging,
because predict.py needs per-patch vectors to say *where* the lesion is.

Usage:
    python embed.py --images data/images --out embeddings.npz
    python embed.py --images data/images --limit 50      # quick smoke test
"""

import argparse
import pathlib

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

MODEL_ID = "facebook/ijepa_vith14_1k"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_images(root, limit=None):
    paths = sorted(
        p for p in pathlib.Path(root).rglob("*") if p.suffix.lower() in IMAGE_EXTS
    )
    if not paths:
        raise SystemExit(f"no images found under {root}")
    return paths[:limit] if limit else paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder of images (searched recursively)")
    ap.add_argument("--out", default="embeddings.npz")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, help="only embed the first N images")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cpu":
        print("  WARNING: no CUDA -- this will be very slow. Install the CUDA build of torch.")

    paths = find_images(args.images, args.limit)
    print(f"found {len(paths)} images")

    print(f"loading {args.model} (first run downloads ~2.5 GB to your HF cache)...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(
        args.model,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()  # frozen: no dropout, no gradients, no training

    batches = []
    for i in range(0, len(paths), args.batch_size):
        chunk = paths[i : i + args.batch_size]
        images = [Image.open(p).convert("RGB") for p in chunk]
        inputs = processor(images, return_tensors="pt").to(device)
        if device == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].half()

        with torch.no_grad():  # frozen -- don't track gradients
            out = model(**inputs)

        # (batch, num_patches, hidden_size) -- one vector per image patch
        batches.append(out.last_hidden_state.to(torch.float16).cpu().numpy())
        print(f"  embedded {min(i + args.batch_size, len(paths))}/{len(paths)}", end="\r")

    embeddings = np.concatenate(batches, axis=0)
    print(f"\nembeddings: {embeddings.shape}  ({embeddings.nbytes / 1e9:.2f} GB in memory)")

    np.savez(args.out, embeddings=embeddings, paths=np.array([str(p) for p in paths]))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
