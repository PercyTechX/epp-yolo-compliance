"""Utilidades geométricas para asociar detecciones EPP-persona.

Todas las cajas se representan como tuplas `(x1, y1, x2, y2)` en
coordenadas absolutas de imagen (píxeles, no normalizadas). Las funciones
son puras, no dependen de Ultralytics ni de OpenCV.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

# Caja en formato xyxy. Documentado como alias para mayor claridad.
BBox = Tuple[float, float, float, float]


def box_area(box: BBox) -> float:
    """Área de una caja xyxy. Retorna 0 si la caja es degenerada."""
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return w * h


def intersection_area(a: BBox, b: BBox) -> float:
    """Área de intersección entre dos cajas xyxy."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    return inter_w * inter_h


def iou(a: BBox, b: BBox) -> float:
    """IoU clásico entre dos cajas xyxy. Retorna 0 si alguna es degenerada."""
    inter = intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def containment(inner: BBox, outer: BBox) -> float:
    """Fracción del área de `inner` que cae dentro de `outer`.

    Útil cuando una caja chica (p.ej. chaleco) debería estar mayormente
    contenida en una caja grande (persona). Devuelve un valor en [0, 1].
    """
    area_inner = box_area(inner)
    if area_inner <= 0.0:
        return 0.0
    return intersection_area(inner, outer) / area_inner


def horizontal_overlap_ratio(a: BBox, b: BBox) -> float:
    """Solapamiento horizontal de `a` sobre `b`, en [0, 1].

    Mide qué fracción del ancho de `a` se proyecta dentro del ancho de `b`.
    Útil para asociar una cabeza a la persona cuyo eje X coincide.
    """
    ax1, _, ax2, _ = a
    bx1, _, bx2, _ = b
    overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    width_a = max(0.0, ax2 - ax1)
    if width_a <= 0.0:
        return 0.0
    return overlap / width_a


def in_upper_third(child: BBox, parent: BBox) -> bool:
    """Determina si el centro vertical de `child` está en el tercio superior de `parent`.

    Heurística para asociación cabeza-persona: la cabeza típicamente cae
    en el tercio superior del bbox de la persona.
    """
    _, py1, _, py2 = parent
    parent_h = py2 - py1
    if parent_h <= 0.0:
        return False
    _, cy1, _, cy2 = child
    child_center_y = (cy1 + cy2) / 2.0
    upper_third_limit = py1 + parent_h / 3.0
    return child_center_y <= upper_third_limit


def associate_head_to_person(
    head: BBox,
    person: BBox,
    min_horizontal_overlap: float = 0.5,
) -> bool:
    """Indica si una cabeza puede asociarse a una persona.

    Reglas (fase 4 del plan):
      1) El centro vertical de la cabeza cae en el tercio superior del bbox
         de la persona.
      2) El solapamiento horizontal (fracción del ancho de la cabeza dentro
         del ancho de la persona) supera el umbral `min_horizontal_overlap`.
    """
    if not in_upper_third(head, person):
        return False
    return horizontal_overlap_ratio(head, person) >= min_horizontal_overlap


def associate_vest_to_person(
    vest: BBox,
    person: BBox,
    min_containment: float = 0.6,
) -> bool:
    """Indica si un chaleco puede asociarse a una persona.

    Regla principal: contención del bbox del chaleco dentro del bbox de la
    persona. Umbral por defecto laxo (0.6) para tolerar cajas mal ajustadas.
    """
    return containment(vest, person) >= min_containment


def best_match_index(
    target: BBox,
    candidates: "list[BBox]",
    score_fn,
    min_score: float,
) -> "int | None":
    """Devuelve el índice del mejor candidato según `score_fn`, o None.

    `score_fn(target, candidate) -> float`. Solo se retorna un índice si
    su score supera `min_score`.
    """
    best_idx: "int | None" = None
    best_score = -np.inf
    for idx, cand in enumerate(candidates):
        score = score_fn(target, cand)
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_idx is None or best_score < min_score:
        return None
    return best_idx
