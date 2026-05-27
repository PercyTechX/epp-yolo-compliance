"""Cuenta cuántas imágenes tienen head + vest detectados simultáneamente.

Diagnóstico del problema de sesgo por dataset: si el modelo casi nunca
predice ambas clases en la misma imagen, el verificador per-trabajador
no puede emitir CONFORME en la mayoría de los casos.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model = YOLO(str(args.weights))

    candidates = []
    for ext in IMAGE_EXTS:
        candidates.extend(args.images.rglob(f"*{ext}"))
    candidates = sorted(candidates)
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    sample = candidates[: args.n]

    stats = {
        "head_helmet_only": 0,
        "head_no_helmet_only": 0,
        "vest_only": 0,
        "head_helmet_and_vest": 0,
        "head_no_helmet_and_vest": 0,
        "any_head_and_vest": 0,
        "no_detections": 0,
    }
    examples_helmet_and_vest = []

    for img_path in sample:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        results = model.predict(image, conf=args.conf, verbose=False)
        result = results[0]
        cls_ids = (
            set()
            if result.boxes is None or len(result.boxes) == 0
            else set(result.boxes.cls.cpu().numpy().astype(int).tolist())
        )

        has_head_helmet = 0 in cls_ids
        has_head_no_helmet = 1 in cls_ids
        has_vest = 3 in cls_ids

        if not cls_ids:
            stats["no_detections"] += 1
        elif has_head_helmet and not has_vest:
            stats["head_helmet_only"] += 1
        elif has_head_no_helmet and not has_vest:
            stats["head_no_helmet_only"] += 1
        elif has_vest and not (has_head_helmet or has_head_no_helmet):
            stats["vest_only"] += 1
        if has_head_helmet and has_vest:
            stats["head_helmet_and_vest"] += 1
            if len(examples_helmet_and_vest) < 10:
                examples_helmet_and_vest.append(img_path.name)
        if has_head_no_helmet and has_vest:
            stats["head_no_helmet_and_vest"] += 1
        if (has_head_helmet or has_head_no_helmet) and has_vest:
            stats["any_head_and_vest"] += 1

    print(f"Sobre {len(sample)} imágenes:")
    for k, v in stats.items():
        print(f"  {k:30s}: {v:4d} ({100*v/len(sample):.1f}%)")
    print("\nEjemplos con head_with_helmet + vest:")
    for ex in examples_helmet_and_vest:
        print(f"  {ex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
