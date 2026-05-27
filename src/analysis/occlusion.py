"""Particiona imágenes por nivel de oclusión a partir de etiquetas YOLO.

Para cada imagen se calcula el solapamiento (IoU) máximo entre cualquier
par de bboxes anotados. Ese valor define el nivel de oclusión de la imagen:

  - Baja  : IoU máximo < `low_threshold`   (default 0.15)
  - Media : low_threshold ≤ IoU < high_threshold
  - Alta  : IoU máximo ≥ `high_threshold`  (default 0.40)

Las funciones operan sobre coordenadas YOLO normalizadas (cx, cy, w, h);
no necesitan abrir las imágenes para calcular IoU porque es invariante a
la escala.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Geometría local (no depende del módulo compliance/ para mantener
# este módulo de análisis autocontenido y reutilizable).
# ---------------------------------------------------------------------------


def _yolo_to_xyxy(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    """Convierte (cx, cy, w, h) normalizado a (x1, y1, x2, y2) normalizado."""
    return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """IoU entre dos cajas xyxy normalizadas."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


# ---------------------------------------------------------------------------
# API pública.
# ---------------------------------------------------------------------------


def read_yolo_bboxes(label_path: Path) -> List[Tuple[float, float, float, float]]:
    """Lee un archivo de etiquetas YOLO y retorna las cajas como xyxy normalizado."""
    bboxes: List[Tuple[float, float, float, float]] = []
    if not label_path.exists():
        return bboxes
    with label_path.open("r", encoding="utf-8") as f:
        for raw in f:
            parts = raw.strip().split()
            if len(parts) < 5:
                continue
            try:
                cx, cy, w, h = (float(parts[i]) for i in range(1, 5))
            except ValueError:
                continue
            bboxes.append(_yolo_to_xyxy(cx, cy, w, h))
    return bboxes


def compute_max_iou(bboxes: List[Tuple[float, float, float, float]]) -> float:
    """Devuelve el IoU máximo entre cualquier par de cajas en la lista.

    Retorna 0.0 si hay menos de 2 cajas.
    """
    n = len(bboxes)
    if n < 2:
        return 0.0
    max_iou = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            iou_ij = _iou(bboxes[i], bboxes[j])
            if iou_ij > max_iou:
                max_iou = iou_ij
    return max_iou


def occlusion_level(max_iou: float, low_threshold: float = 0.15, high_threshold: float = 0.40) -> str:
    """Clasifica un IoU máximo en 'low' / 'medium' / 'high'."""
    if max_iou < low_threshold:
        return "low"
    if max_iou < high_threshold:
        return "medium"
    return "high"


def partition_by_occlusion(
    labels_dir: Path,
    low_threshold: float = 0.15,
    high_threshold: float = 0.40,
) -> Dict[str, List[str]]:
    """Particiona los stems (nombres sin extensión) de las etiquetas por nivel de oclusión.

    Devuelve un dict con claves 'low', 'medium', 'high'. El valor es la lista
    de stems (sin extensión y sin ruta) de las imágenes correspondientes.
    Las etiquetas vacías o sin caja se ignoran (sin ground truth no se puede
    evaluar la imagen).
    """
    partitions: Dict[str, List[str]] = {"low": [], "medium": [], "high": []}
    for label_file in sorted(labels_dir.rglob("*.txt")):
        bboxes = read_yolo_bboxes(label_file)
        if not bboxes:
            continue
        max_iou = compute_max_iou(bboxes)
        level = occlusion_level(max_iou, low_threshold, high_threshold)
        partitions[level].append(label_file.stem)
    return partitions
