"""Dibuja bboxes sobre una muestra aleatoria de imágenes para validación visual.

Útil tras `remap_labels.py` para verificar manualmente que los ids
unificados se correspondan con lo que se ve en la imagen.

Las imágenes anotadas se escriben en `--out`, no se muestran en pantalla
(funciona en entornos sin GUI). El operador las revisa después.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING = REPO_ROOT / "configs" / "class_mapping.yaml"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Colores BGR fijos por clase para inspección consistente.
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 200, 0),       # head_with_helmet     -> verde
    1: (0, 0, 220),       # head_without_helmet  -> rojo
    2: (220, 180, 0),     # person               -> cian
    3: (0, 200, 200),     # vest                 -> amarillo
    4: (140, 0, 200),     # no_vest_person       -> púrpura
}


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Renderiza bboxes sobre N imágenes aleatorias para validar "
            "manualmente el remapeo."
        )
    )
    parser.add_argument("--src", required=True, type=Path, help="Dataset con images/ y labels/.")
    parser.add_argument("--out", required=True, type=Path, help="Directorio destino de imágenes anotadas.")
    parser.add_argument("--n", type=int, default=100, help="Número de imágenes a anotar.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria.")
    parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_MAPPING,
        help="class_mapping.yaml para nombres legibles (no obligatorio).",
    )
    return parser.parse_args()


def load_class_names(mapping_path: Path) -> Dict[int, str]:
    """Carga los nombres de clase de la taxonomía unificada."""
    if not mapping_path.exists():
        return {}
    with mapping_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    unified = cfg.get("unified_classes", {})
    return {int(k): v for k, v in unified.items()}


def yolo_to_xyxy(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> Tuple[int, int, int, int]:
    """Convierte (cx, cy, w, h) normalizado a (x1, y1, x2, y2) en píxeles."""
    x1 = int((cx - w / 2.0) * img_w)
    y1 = int((cy - h / 2.0) * img_h)
    x2 = int((cx + w / 2.0) * img_w)
    y2 = int((cy + h / 2.0) * img_h)
    return x1, y1, x2, y2


def render_image(
    img_path: Path, label_path: Path, names: Dict[int, str]
) -> "cv2.typing.MatLike | None":  # type: ignore[name-defined]
    """Lee una imagen y dibuja sus bboxes; retorna None si no se pudo leer."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    if label_path.exists():
        with label_path.open("r", encoding="utf-8") as f:
            for raw in f:
                parts = raw.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                except ValueError:
                    continue
                x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, w, h)
                color = CLASS_COLORS.get(cid, (200, 200, 200))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = names.get(cid, str(cid))
                cv2.putText(
                    img,
                    label,
                    (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
    return img


def main() -> int:
    """Punto de entrada principal."""
    args = parse_args()
    images_root = args.src / "images"
    labels_root = args.src / "labels"
    if not images_root.exists() or not labels_root.exists():
        print(f"[validate][error] Estructura esperada no encontrada en {args.src}.", file=sys.stderr)
        return 2

    names = load_class_names(args.mapping)

    candidates: List[Path] = []
    for ext in IMAGE_EXTS:
        candidates.extend(images_root.rglob(f"*{ext}"))
    if not candidates:
        print(f"[validate][error] No se encontraron imágenes bajo {images_root}.", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    sample = candidates[: args.n]

    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    for img_path in sample:
        rel = img_path.relative_to(images_root).with_suffix(".txt")
        lbl_path = labels_root / rel
        rendered = render_image(img_path, lbl_path, names)
        if rendered is None:
            continue
        out_name = "__".join(img_path.with_suffix("").parts[-3:]) + img_path.suffix.lower()
        out_path = args.out / out_name
        cv2.imwrite(str(out_path), rendered)
        written += 1

    print(f"[validate] OK: {written} imágenes anotadas en {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
