"""Renderiza el resultado del verificador sobre una imagen.

Dos niveles de información por trabajador:
  1. Bbox grande del cuerpo, coloreado por veredicto
     (verde CONFORME, rojo NO_CONFORME, amarillo INDETERMINADO).
  2. Bbox chico de la cabeza con la confianza del detector
     ("Casco XX%" o "Sin casco XX%").

Al pie del frame: barra de contadores acumulada del frame actual con
total de trabajadores detectados, conteo por estado y tasa de
cumplimiento.

El módulo acepta una imagen `numpy.ndarray` BGR (formato OpenCV) y un
`FrameReport`, y retorna la misma imagen anotada.
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

from src.compliance.verifier import (
    CLASS_HEAD_WITH_HELMET,
    CLASS_HEAD_WITHOUT_HELMET,
    ComplianceStatus,
    FrameReport,
    PersonVerdict,
)

# Colores BGR (OpenCV) por estado.
COLOR_CONFORME = (0, 200, 0)        # verde
COLOR_NO_CONFORME = (0, 0, 220)     # rojo
COLOR_INDETERMINADO = (0, 220, 220) # amarillo
COLOR_TEXT_WHITE = (255, 255, 255)
COLOR_BAR_BG = (20, 20, 20)         # casi negro


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

    `image` debe ser BGR uint8. Se trabaja sobre una copia.
    `draw_rate=False` desactiva la barra inferior con contadores.
    """
    out = image.copy()
    for verdict in report.verdicts:
        _draw_body_bbox(out, verdict)
        _draw_head_bbox(out, verdict)
    if draw_rate:
        _draw_counter_bar(out, report)
    return out


def _draw_body_bbox(image: np.ndarray, verdict: PersonVerdict) -> None:
    """Dibuja el bbox grande del cuerpo con la etiqueta del veredicto."""
    x1, y1, x2, y2 = (int(c) for c in verdict.person_box)
    color = _color_for(verdict.status)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    label = _label_for_body(verdict)
    _draw_label_above(image, label, (x1, y1), color)


def _draw_head_bbox(image: np.ndarray, verdict: PersonVerdict) -> None:
    """Dibuja el bbox chico de la cabeza, solo con el color del veredicto.

    El bbox de la cabeza no lleva etiqueta de texto para evitar solape
    con la etiqueta del cuerpo. Su única función visual es señalar
    dónde el detector encontró la cabeza, reforzando con color el
    veredicto del trabajador completo.
    """
    if verdict.head_box is None or verdict.head_class is None:
        return
    hx1, hy1, hx2, hy2 = (int(c) for c in verdict.head_box)
    color = _color_for(verdict.status)
    cv2.rectangle(image, (hx1, hy1), (hx2, hy2), color, 2)


def _label_for_body(verdict: PersonVerdict) -> str:
    """Etiqueta corta para el bbox del cuerpo según el veredicto."""
    if verdict.status == ComplianceStatus.CONFORME:
        return "CONFORME"
    if verdict.status == ComplianceStatus.NO_CONFORME:
        if verdict.reasons:
            return "NO_CONFORME: " + ", ".join(verdict.reasons)
        return "NO_CONFORME"
    return "INDETERMINADO"


def _label_for_head(verdict: PersonVerdict) -> str:
    """Etiqueta de la cabeza: tipo de detección + confianza."""
    if verdict.head_class == CLASS_HEAD_WITH_HELMET:
        base = "Casco"
    elif verdict.head_class == CLASS_HEAD_WITHOUT_HELMET:
        base = "Sin casco"
    else:
        return ""
    if verdict.head_score is not None:
        return f"{base} {verdict.head_score * 100:.0f}%"
    return base


def _draw_label_above(
    image: np.ndarray,
    text: str,
    origin_xy: "tuple[int, int]",
    color: "tuple[int, int, int]",
    font_scale: float = 0.5,
    thickness: int = 1,
) -> None:
    """Dibuja una etiqueta con fondo coloreado encima del origin (esquina superior izq del bbox)."""
    x, y = origin_xy
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    margin = 3
    box_x1 = x
    box_y1 = max(0, y - th - 2 * margin)
    box_x2 = x + tw + 2 * margin
    box_y2 = y
    cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), color, thickness=-1)
    cv2.putText(
        image,
        text,
        (box_x1 + margin, box_y2 - margin - baseline // 2),
        font,
        font_scale,
        COLOR_TEXT_WHITE,
        thickness,
        cv2.LINE_AA,
    )


def _draw_counter_bar(image: np.ndarray, report: FrameReport) -> None:
    """Dibuja la barra inferior con contadores y tasa de cumplimiento."""
    h, w = image.shape[:2]
    bar_height = 42
    overlay = image.copy()
    cv2.rectangle(overlay, (0, h - bar_height), (w, h), COLOR_BAR_BG, thickness=-1)
    alpha = 0.65
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Contadores agregados.
    n_total = len(report.verdicts)
    n_conf = sum(1 for v in report.verdicts if v.status == ComplianceStatus.CONFORME)
    n_no = sum(1 for v in report.verdicts if v.status == ComplianceStatus.NO_CONFORME)
    n_ind = sum(1 for v in report.verdicts if v.status == ComplianceStatus.INDETERMINADO)
    rate_text = f"{report.compliance_rate * 100:.0f}%"

    # Construir segmentos con colores propios para que cada número se pinte
    # con el color de su estado.
    segments: List["tuple[str, tuple[int, int, int]]"] = [
        (f"Detectados: {n_total}", COLOR_TEXT_WHITE),
        (f"CONFORME: {n_conf}", COLOR_CONFORME),
        (f"NO_CONFORME: {n_no}", COLOR_NO_CONFORME),
        (f"INDETERMINADO: {n_ind}", COLOR_INDETERMINADO),
        (f"Cumplimiento: {rate_text}", COLOR_TEXT_WHITE),
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    text_y = h - 14

    # Centrar horizontalmente los segmentos con separadores entre ellos.
    sep = "  |  "
    widths = [
        cv2.getTextSize(t, font, font_scale, thickness)[0][0] for t, _ in segments
    ]
    sep_width = cv2.getTextSize(sep, font, font_scale, thickness)[0][0]
    total_width = sum(widths) + sep_width * (len(segments) - 1)
    x = max(16, (w - total_width) // 2)

    for i, (text, color) in enumerate(segments):
        cv2.putText(
            image,
            text,
            (x, text_y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        x += widths[i]
        if i < len(segments) - 1:
            cv2.putText(
                image,
                sep,
                (x, text_y),
                font,
                font_scale,
                COLOR_TEXT_WHITE,
                thickness,
                cv2.LINE_AA,
            )
            x += sep_width
