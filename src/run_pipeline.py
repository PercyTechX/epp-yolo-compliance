"""Pipeline completo de inferencia: YOLO + verificador determinista.

Toma un checkpoint `.pt` entrenado y un directorio de imágenes, corre
detección con Ultralytics, convierte las detecciones al tipo `Detection`
del verificador, evalúa cumplimiento y guarda las imágenes anotadas con
bboxes verde (CONFORME), rojo (NO_CONFORME) y amarillo (INDETERMINADO).

Uso típico:
    python -m src.run_pipeline \
        --weights runs_outputs/exp_D/best.pt \
        --images data/processed/images/val \
        --out reports/figures/verifier_demo_exp_D \
        --n 20 \
        --conf 0.25

Pensado para revisión visual del Fase 4. Para video, usar el notebook
06_inference_demo.ipynb en Kaggle.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List

import cv2

# Importaciones de Ultralytics y del módulo verificador del proyecto.
from ultralytics import YOLO

from src.compliance.renderer import annotate_frame
from src.compliance.verifier import Detection, verify_frame


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Aplica YOLO + verificador determinista a una muestra de imágenes "
            "y guarda los resultados anotados."
        )
    )
    parser.add_argument(
        "--weights",
        required=True,
        type=Path,
        help="Ruta al .pt entrenado (ej: runs_outputs/exp_D/best.pt).",
    )
    parser.add_argument(
        "--images",
        required=True,
        type=Path,
        help="Directorio con imágenes a procesar (recursivo).",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Directorio destino para las imágenes anotadas.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Número de imágenes a procesar (muestra aleatoria).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Umbral de confianza para YOLO (default 0.25 = default Ultralytics).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="Umbral de IoU para NMS (default 0.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para la muestra.",
    )
    return parser.parse_args()


def collect_images(root: Path) -> List[Path]:
    """Recolecta recursivamente todas las imágenes bajo `root`."""
    found: List[Path] = []
    for ext in IMAGE_EXTS:
        found.extend(root.rglob(f"*{ext}"))
    return sorted(found)


def yolo_result_to_detections(result) -> List[Detection]:
    """Convierte un objeto Results de Ultralytics a una lista de `Detection`."""
    detections: List[Detection] = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    for box, cid, conf in zip(boxes_xyxy, cls_ids, confs):
        detections.append(
            Detection(
                class_id=int(cid),
                box=tuple(float(v) for v in box.tolist()),
                score=float(conf),
            )
        )
    return detections


def main() -> int:
    """Punto de entrada principal."""
    args = parse_args()

    if not args.weights.exists():
        print(f"[pipeline][error] No existe el checkpoint: {args.weights}", file=sys.stderr)
        return 2
    if not args.images.exists():
        print(f"[pipeline][error] No existe el directorio de imágenes: {args.images}", file=sys.stderr)
        return 2

    print(f"[pipeline] Cargando modelo: {args.weights}")
    model = YOLO(str(args.weights))

    print(f"[pipeline] Recolectando imágenes en: {args.images}")
    candidates = collect_images(args.images)
    if not candidates:
        print("[pipeline][error] No se encontraron imágenes.", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    sample = candidates[: args.n]
    print(f"[pipeline] Procesando {len(sample)} imágenes (de {len(candidates)} totales).")

    args.out.mkdir(parents=True, exist_ok=True)

    # Estadísticas agregadas para reporte rápido.
    total_conformes = 0
    total_no_conformes = 0
    total_indeterminados = 0

    for i, img_path in enumerate(sample, 1):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[pipeline][warn] No se pudo leer {img_path}, salto.")
            continue

        results = model.predict(
            image,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )
        result = results[0]
        detections = yolo_result_to_detections(result)
        report = verify_frame(detections)

        annotated = annotate_frame(image, report)
        # Nombre de salida: índice + nombre base, para mantener orden de revisión.
        out_name = f"{i:03d}__{img_path.stem}.jpg"
        cv2.imwrite(str(args.out / out_name), annotated)

        # Actualizar conteos.
        for v in report.verdicts:
            if v.status.value == "CONFORME":
                total_conformes += 1
            elif v.status.value == "NO_CONFORME":
                total_no_conformes += 1
            else:
                total_indeterminados += 1

        if i % 5 == 0 or i == len(sample):
            print(f"[pipeline]   {i}/{len(sample)} imágenes procesadas")

    total = total_conformes + total_no_conformes + total_indeterminados
    print()
    print(f"[pipeline] OK. Resultados en {args.out}")
    print(f"[pipeline] Veredictos totales: {total}")
    print(f"[pipeline]   CONFORME       : {total_conformes}")
    print(f"[pipeline]   NO_CONFORME    : {total_no_conformes}")
    print(f"[pipeline]   INDETERMINADO  : {total_indeterminados}")
    if total > 0:
        rate = total_conformes / max(1, total_conformes + total_no_conformes)
        print(f"[pipeline] Tasa de cumplimiento (excl. INDETERMINADO): {rate*100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
