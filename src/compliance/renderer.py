"""Renderiza el resultado del verificador sobre una imagen.

Bboxes verdes para CONFORME, rojos para NO_CONFORME, amarillos para
INDETERMINADO. Usa Supervision para la anotación (más declarativo que
OpenCV puro, según el archivo 02 del stack).

El módulo expone una API que acepta una imagen como `numpy.ndarray` BGR
(formato OpenCV) y un `FrameReport`, y retorna la misma imagen anotada.
"""
from __future__ import annotations

from typing import List

import numpy as np

try:
    import supervision as sv  # type: ignore
except ImportError:  # pragma: no cover
    sv = None  # Permitimos importar el módulo aun sin supervision instalado.

from src.compliance.verifier import (
    ComplianceStatus,
    FrameReport,
    PersonVerdict,
)

# Colores BGR (OpenCV) por estado.
COLOR_CONFORME = (0, 200, 0)        # verde
COLOR_NO_CONFORME = (0, 0, 220)     # rojo
COLOR_INDETERMINADO = (0, 220, 220) # amarillo


def _color_for(status: ComplianceStatus) -> "tuple[int, int, int]":
    """Retorna el color BGR asociado a un estado."""
    if status == ComplianceStatus.CONFORME:
        return COLOR_CONFORME
    if status == ComplianceStatus.NO_CONFORME:
        return COLOR_NO_CONFORME
    return COLOR_INDETERMINADO


def annotate_frame(
    image: np.ndarray,
    report: FrameReport,
    draw_rate: bool = True,
) -> np.ndarray:
    """Dibuja sobre `image` los veredictos del reporte.

    Si `supervision` está disponible se usa para los anotadores; en caso
    contrario se cae a OpenCV puro como fallback (mantiene el módulo
    utilizable en entornos donde supervision no esté instalado).

    `image` debe ser BGR uint8. La función trabaja sobre una copia.
    """
    out = image.copy()
    if sv is not None:
        return _annotate_with_supervision(out, report, draw_rate)
    return _annotate_with_opencv(out, report, draw_rate)


def _annotate_with_supervision(
    image: np.ndarray, report: FrameReport, draw_rate: bool
) -> np.ndarray:
    """Anotación basada en supervision.Detections + anotadores de la librería."""
    # Construimos una Detections por estado para colorearlas distinto.
    by_status: "dict[ComplianceStatus, List[PersonVerdict]]" = {
        ComplianceStatus.CONFORME: [],
        ComplianceStatus.NO_CONFORME: [],
        ComplianceStatus.INDETERMINADO: [],
    }
    for v in report.verdicts:
        by_status[v.status].append(v)

    for status, verdicts in by_status.items():
        if not verdicts:
            continue
        xyxy = np.array([v.person_box for v in verdicts], dtype=float)
        detections = sv.Detections(xyxy=xyxy)
        color_bgr = _color_for(status)
        color = sv.Color(r=color_bgr[2], g=color_bgr[1], b=color_bgr[0])
        box_annotator = sv.BoxAnnotator(color=color, thickness=2)
        image = box_annotator.annotate(scene=image, detections=detections)

        label_annotator = sv.LabelAnnotator(color=color, text_color=sv.Color.WHITE)
        labels = [_label_for(v) for v in verdicts]
        image = label_annotator.annotate(
            scene=image, detections=detections, labels=labels
        )

    if draw_rate:
        _draw_rate(image, report.compliance_rate)
    return image


def _annotate_with_opencv(
    image: np.ndarray, report: FrameReport, draw_rate: bool
) -> np.ndarray:
    """Anotación mínima con OpenCV puro como fallback."""
    import cv2  # importación local para no obligar a OpenCV en tests unitarios

    for v in report.verdicts:
        x1, y1, x2, y2 = (int(c) for c in v.person_box)
        color = _color_for(v.status)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            _label_for(v),
            (x1, max(y1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    if draw_rate:
        _draw_rate(image, report.compliance_rate)
    return image


def _label_for(verdict: PersonVerdict) -> str:
    """Texto corto para etiquetar un bbox según su veredicto."""
    if verdict.status == ComplianceStatus.CONFORME:
        return "CONFORME"
    if verdict.status == ComplianceStatus.NO_CONFORME:
        if verdict.reasons:
            return "NO_CONFORME: " + ", ".join(verdict.reasons)
        return "NO_CONFORME"
    return "INDETERMINADO"


def _draw_rate(image: np.ndarray, rate: float) -> None:
    """Estampa la tasa de cumplimiento del fotograma en la esquina superior izquierda."""
    import cv2

    text = f"Cumplimiento: {rate * 100:5.1f}%"
    cv2.rectangle(image, (8, 8), (260, 38), (0, 0, 0), thickness=-1)
    cv2.putText(
        image,
        text,
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
