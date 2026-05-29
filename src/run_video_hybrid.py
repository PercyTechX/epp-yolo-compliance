"""Pipeline híbrido: 2 modelos YOLO + verificador.

Combina:
  - Modelo D (nuestro entrenamiento) -> aporta solo cabezas con/sin casco.
  - YOLOv8s pre-entrenado en COCO   -> aporta solo `person` (clase 0 de COCO).

Las detecciones de ambos se unen en una lista única que se pasa al
verificador determinista. El verificador y el renderer no cambian.

Motivación: nuestro modelo D tiene desempeño excelente para cabezas con
casco (mAP@0.5 = 0.97) pero pobre para `person` (mAP@0.5 = 0.17, escasez
de datos en los datasets que combinamos). YOLOv8s COCO está entrenado
sobre >200.000 imágenes con millones de personas anotadas en todas las
poses; su mAP@0.5 para `person` es ~0.85.

El resultado esperado: cuando un trabajador es trivialmente visible
para el ojo humano (incluso agachado, de espalda o en close-up
inusual), el sistema lo reconoce al menos como persona y emite
INDETERMINADO si no se ve la cabeza. Esto evita "frames vacíos" que
violarían el principio "si lo ve el humano, debería verlo la IA".

Uso típico:
    python -m src.run_video_hybrid \
        --weights-helmet runs_outputs/exp_D/best.pt \
        --video demo_video/video03p.mp4 \
        --out demo_video/video03_hybrid.mp4 \
        --conf-helmet 0.15 --conf-person 0.40 --imgsz 1280
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import cv2

from ultralytics import YOLO

from src.compliance.renderer import annotate_frame
from src.compliance.verifier import (
    CLASS_HEAD_WITH_HELMET,
    CLASS_HEAD_WITHOUT_HELMET,
    CLASS_PERSON,
    Detection,
    verify_frame,
)

# En COCO la clase `person` es la 0.
COCO_PERSON_CLASS = 0


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline híbrido: modelo D para casco + COCO pre-entrenado "
            "para detección robusta de personas + verificador."
        )
    )
    parser.add_argument(
        "--weights-helmet",
        required=True,
        type=Path,
        help="Ruta al best.pt entrenado (modelo D).",
    )
    parser.add_argument(
        "--weights-coco",
        default="yolov8s.pt",
        type=str,
        help=(
            "Pesos de YOLOv8 COCO. Si es un nombre sin ruta, Ultralytics "
            "lo descarga automáticamente la primera vez."
        ),
    )
    parser.add_argument(
        "--video",
        required=True,
        type=Path,
        help="Video de entrada.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Video anotado de salida.",
    )
    parser.add_argument(
        "--conf-helmet",
        type=float,
        default=0.15,
        help="Umbral de confianza para el modelo de casco.",
    )
    parser.add_argument(
        "--conf-person",
        type=float,
        default=0.40,
        help="Umbral de confianza para el detector COCO de personas.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="Umbral IoU para NMS en ambos modelos.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Tamaño de inferencia.",
    )
    return parser.parse_args()


def helmet_results_to_detections(result) -> List[Detection]:
    """Convierte salidas del modelo D a `Detection`, conservando solo cabezas."""
    dets: List[Detection] = []
    if result.boxes is None or len(result.boxes) == 0:
        return dets
    xyxy = result.boxes.xyxy.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    for box, cid, conf in zip(xyxy, cls_ids, confs):
        cid_int = int(cid)
        # Solo retenemos cabezas; ignoramos vest, no_vest_person y person
        # del modelo D (este último viene del modelo COCO).
        if cid_int in (CLASS_HEAD_WITH_HELMET, CLASS_HEAD_WITHOUT_HELMET):
            dets.append(
                Detection(
                    class_id=cid_int,
                    box=tuple(float(v) for v in box.tolist()),
                    score=float(conf),
                )
            )
    return dets


def coco_results_to_person_detections(result) -> List[Detection]:
    """Convierte salidas del modelo COCO a `Detection` de clase `person`.

    Solo se procesan detecciones con `class_id == COCO_PERSON_CLASS`.
    Se remapean a la clase 2 (`person`) de la taxonomía unificada del
    proyecto.
    """
    dets: List[Detection] = []
    if result.boxes is None or len(result.boxes) == 0:
        return dets
    xyxy = result.boxes.xyxy.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    for box, cid, conf in zip(xyxy, cls_ids, confs):
        if int(cid) == COCO_PERSON_CLASS:
            dets.append(
                Detection(
                    class_id=CLASS_PERSON,
                    box=tuple(float(v) for v in box.tolist()),
                    score=float(conf),
                )
            )
    return dets


def main() -> int:
    """Punto de entrada principal."""
    args = parse_args()

    if not args.weights_helmet.exists():
        print(f"[hybrid][error] No existe el checkpoint de casco: {args.weights_helmet}", file=sys.stderr)
        return 2
    if not args.video.exists():
        print(f"[hybrid][error] No existe el video: {args.video}", file=sys.stderr)
        return 2

    print(f"[hybrid] Cargando modelo de casco : {args.weights_helmet}")
    model_helmet = YOLO(str(args.weights_helmet))

    print(f"[hybrid] Cargando modelo COCO     : {args.weights_coco}")
    print("[hybrid]   (se descarga ~22 MB la primera vez)")
    model_coco = YOLO(args.weights_coco)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"[hybrid][error] No se pudo abrir {args.video}", file=sys.stderr)
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[hybrid] {width}x{height} @ {fps:.1f} fps, {total} frames")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out), fourcc, fps, (width, height))

    totals = {"CONFORME": 0, "NO_CONFORME": 0, "INDETERMINADO": 0}
    start = time.time()
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # Inferencia con el modelo de casco.
        res_helmet = model_helmet.predict(
            frame, conf=args.conf_helmet, iou=args.iou, imgsz=args.imgsz, verbose=False
        )
        # Inferencia con el modelo COCO solo para `person`.
        res_coco = model_coco.predict(
            frame,
            conf=args.conf_person,
            iou=args.iou,
            imgsz=args.imgsz,
            classes=[COCO_PERSON_CLASS],
            verbose=False,
        )

        dets = (
            helmet_results_to_detections(res_helmet[0])
            + coco_results_to_person_detections(res_coco[0])
        )
        report = verify_frame(dets)
        annotated = annotate_frame(frame, report)
        writer.write(annotated)

        for v in report.verdicts:
            totals[v.status.value] += 1

        if frame_idx % 30 == 0:
            elapsed = time.time() - start
            print(f"[hybrid]   {frame_idx}/{total} ({elapsed:.0f}s)")

    cap.release()
    writer.release()
    elapsed = time.time() - start

    print()
    print(f"[hybrid] OK. Video anotado: {args.out}")
    print(f"[hybrid] Tiempo total: {elapsed:.0f}s ({frame_idx} frames)")
    print(f"[hybrid] Veredictos acumulados: {totals}")
    determinate = totals["CONFORME"] + totals["NO_CONFORME"]
    if determinate > 0:
        rate = totals["CONFORME"] / determinate * 100
        print(f"[hybrid] Tasa de cumplimiento (excl. INDETERMINADO): {rate:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
