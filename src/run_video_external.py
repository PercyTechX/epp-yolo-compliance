"""Pipeline de video usando un modelo YOLO externo (OtroEnfoque).

Wrapper que carga un checkpoint con una taxonomía diferente, remapea los
ids de clase a la taxonomía unificada del proyecto y luego pasa las
detecciones al verificador determinista existente.

Caso de uso: comparar nuestro modelo D (entrenado con datasets de
anotación parcial) contra un modelo entrenado sobre un dataset con
anotación completa por trabajador.

Uso:
    python -m src.run_video_external \
        --weights OtroEnfoque/detection_ppe/prediction_module/weights/best.pt \
        --video demo_video/video03p.mp4 \
        --out demo_video/video03_otroenfoque.mp4 \
        --conf 0.25 --imgsz 1280

Mapeo de clases (10 -> 5):
    0 Hardhat        -> 0 head_with_helmet
    1 Mask           -> (descartar)
    2 NO-Hardhat     -> 1 head_without_helmet
    3 NO-Mask        -> (descartar)
    4 NO-Safety Vest -> 4 no_vest_person
    5 Person         -> 2 person
    6 Safety Cone    -> (descartar)
    7 Safety Vest    -> 3 vest
    8 machinery      -> (descartar)
    9 vehicle        -> (descartar)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from ultralytics import YOLO

from src.compliance.renderer import annotate_frame
from src.compliance.verifier import Detection, verify_frame


# Tabla de mapeo del modelo OtroEnfoque (10 clases) a nuestra taxonomía (5 clases).
# Las clases con valor None se descartan (no aportan al verificador).
CLASS_REMAP_OTROENFOQUE: Dict[int, Optional[int]] = {
    0: 0,    # Hardhat        -> head_with_helmet
    1: None, # Mask           -> descartar
    2: 1,    # NO-Hardhat     -> head_without_helmet
    3: None, # NO-Mask        -> descartar
    4: 4,    # NO-Safety Vest -> no_vest_person
    5: 2,    # Person         -> person
    6: None, # Safety Cone    -> descartar
    7: 3,    # Safety Vest    -> vest
    8: None, # machinery      -> descartar
    9: None, # vehicle        -> descartar
}


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline YOLO + verificador usando un modelo externo con "
            "taxonomía distinta. Las clases se remapean a la taxonomía "
            "unificada antes de pasar al verificador."
        )
    )
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def remap_detections(result, class_remap: Dict[int, Optional[int]]) -> List[Detection]:
    """Convierte un Results de Ultralytics + tabla de remap a `Detection`."""
    dets: List[Detection] = []
    if result.boxes is None or len(result.boxes) == 0:
        return dets
    xyxy = result.boxes.xyxy.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    for box, cid, conf in zip(xyxy, cls_ids, confs):
        new_cid = class_remap.get(int(cid))
        if new_cid is None:
            continue
        dets.append(
            Detection(
                class_id=int(new_cid),
                box=tuple(float(v) for v in box.tolist()),
                score=float(conf),
            )
        )
    return dets


def main() -> int:
    """Punto de entrada principal."""
    args = parse_args()

    if not args.weights.exists():
        print(f"[ext][error] No existe el checkpoint: {args.weights}", file=sys.stderr)
        return 2
    if not args.video.exists():
        print(f"[ext][error] No existe el video: {args.video}", file=sys.stderr)
        return 2

    print(f"[ext] Cargando modelo externo: {args.weights}")
    model = YOLO(str(args.weights))
    print(f"[ext] Clases del modelo: {model.names}")
    print(f"[ext] Mapeo aplicado: {CLASS_REMAP_OTROENFOQUE}")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"[ext][error] No se pudo abrir {args.video}", file=sys.stderr)
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[ext] {w}x{h} @ {fps:.1f} fps, {total} frames")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out), fourcc, fps, (w, h))

    totals = {"CONFORME": 0, "NO_CONFORME": 0, "INDETERMINADO": 0}
    start = time.time()
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        results = model.predict(
            frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False
        )
        dets = remap_detections(results[0], CLASS_REMAP_OTROENFOQUE)
        report = verify_frame(dets)
        annotated = annotate_frame(frame, report)
        for v in report.verdicts:
            totals[v.status.value] += 1
        writer.write(annotated)

        if frame_idx % 30 == 0:
            elapsed = time.time() - start
            print(f"[ext]   {frame_idx}/{total} ({elapsed:.0f}s)")

    cap.release()
    writer.release()
    elapsed = time.time() - start

    print()
    print(f"[ext] OK. Video anotado: {args.out}")
    print(f"[ext] Tiempo total: {elapsed:.0f}s ({frame_idx} frames)")
    print(f"[ext] Veredictos: {totals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
