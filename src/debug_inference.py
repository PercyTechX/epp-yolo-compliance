"""Diagnóstico rápido: imprime detecciones crudas + veredictos para N imágenes.

Permite ver qué clase está detectando el modelo y por qué el verificador
toma una decisión u otra.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.compliance.verifier import Detection, verify_frame

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_NAMES = {
    0: "head_with_helmet",
    1: "head_without_helmet",
    2: "person",
    3: "vest",
    4: "no_vest_person",
}


def main() -> int:
    """Punto de entrada."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--n", type=int, default=5)
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

    for i, img_path in enumerate(sample, 1):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        results = model.predict(image, conf=args.conf, verbose=False)
        result = results[0]

        print(f"\n=== Imagen {i}: {img_path.name} ({w}x{h}) ===")
        if result.boxes is None or len(result.boxes) == 0:
            print("  Sin detecciones.")
            continue

        dets = []
        for box, cid, conf in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.cls.cpu().numpy().astype(int),
            result.boxes.conf.cpu().numpy(),
        ):
            x1, y1, x2, y2 = [int(v) for v in box]
            name = CLASS_NAMES.get(int(cid), str(cid))
            print(f"  {name:22s} conf={conf:.2f} bbox=({x1},{y1},{x2},{y2}) size={x2-x1}x{y2-y1}")
            dets.append(
                Detection(class_id=int(cid), box=tuple(box.tolist()), score=float(conf))
            )

        report = verify_frame(dets)
        print(f"  --- Verificador: {len(report.verdicts)} veredictos, tasa={report.compliance_rate*100:.0f}% ---")
        for v in report.verdicts:
            head_info = f"head_class={v.head_class}" if v.head_class is not None else "sin_cabeza"
            print(f"    {v.status.value:14s} {head_info:25s} reasons={v.reasons}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
