"""Pipeline YOLO + verificador sobre un video corto, frame a frame.

Toma un .mp4, corre detección con Ultralytics sobre cada fotograma,
aplica el verificador determinista y exporta el video anotado con
bboxes verde / rojo / amarillo + tasa de cumplimiento por fotograma.

Uso típico (local, CPU):
    python -m src.run_video \
        --weights runs_outputs/exp_D/best.pt \
        --video demo_video/video02.mp4 \
        --out demo_video/video02_annotated.mp4 \
        --conf 0.25

Sobre 790 fotogramas en CPU el procesamiento toma ~5-10 minutos.
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
from src.compliance.verifier import Detection, verify_frame


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Procesa un video corto con YOLO + verificador y exporta el "
            "video anotado con bboxes coloreados por estado."
        )
    )
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Umbral de confianza para YOLO."
    )
    parser.add_argument(
        "--iou", type=float, default=0.5, help="Umbral IoU para NMS."
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Tamaño de inferencia. Subir a 960/1280 ayuda con objetos pequeños.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Si > 1, procesa 1 cada N fotogramas (acelera el run).",
    )
    return parser.parse_args()


def yolo_to_detections(result) -> List[Detection]:
    """Convierte un Results de Ultralytics a una lista de `Detection`."""
    dets: List[Detection] = []
    if result.boxes is None or len(result.boxes) == 0:
        return dets
    xyxy = result.boxes.xyxy.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    for box, cid, conf in zip(xyxy, cls_ids, confs):
        dets.append(
            Detection(
                class_id=int(cid),
                box=tuple(float(v) for v in box.tolist()),
                score=float(conf),
            )
        )
    return dets


def main() -> int:
    """Punto de entrada principal."""
    args = parse_args()

    if not args.weights.exists():
        print(f"[video][error] No existe el checkpoint: {args.weights}", file=sys.stderr)
        return 2
    if not args.video.exists():
        print(f"[video][error] No existe el video: {args.video}", file=sys.stderr)
        return 2

    print(f"[video] Cargando modelo: {args.weights}")
    model = YOLO(str(args.weights))

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"[video][error] No se pudo abrir {args.video}", file=sys.stderr)
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video] {width}x{height} @ {fps:.1f} fps, {total} frames")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out), fourcc, fps, (width, height))

    # Estadísticas agregadas.
    total_verdicts = {"CONFORME": 0, "NO_CONFORME": 0, "INDETERMINADO": 0}
    last_annotated = None
    start = time.time()
    frame_idx = 0
    processed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # Decide si procesamos este fotograma o repetimos el último anotado.
        if (frame_idx - 1) % args.frame_skip == 0:
            results = model.predict(
                frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False
            )
            dets = yolo_to_detections(results[0])
            report = verify_frame(dets)
            annotated = annotate_frame(frame, report)
            for v in report.verdicts:
                total_verdicts[v.status.value] += 1
            last_annotated = annotated
            processed += 1
        else:
            # Repetimos la última anotación para mantener fluidez visual sin
            # gastar GPU/CPU innecesariamente.
            annotated = last_annotated if last_annotated is not None else frame

        writer.write(annotated)

        if frame_idx % 30 == 0:
            elapsed = time.time() - start
            print(f"[video]   {frame_idx}/{total} frames ({elapsed:.0f}s)")

    cap.release()
    writer.release()

    elapsed = time.time() - start
    print()
    print(f"[video] OK. Video anotado: {args.out} ({args.out.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"[video] Tiempo total: {elapsed:.0f}s ({frame_idx} frames, {processed} con inferencia)")
    print(f"[video] Veredictos acumulados: {total_verdicts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
