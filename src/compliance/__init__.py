"""Módulo verificador de cumplimiento de EPP.

Consume detecciones YOLO por fotograma y emite un veredicto determinista
por persona detectada (CONFORME / NO_CONFORME / INDETERMINADO).

API pública:
    - geometry: utilidades geométricas puras (IoU, contención, asociación).
    - verifier: lógica de cumplimiento por fotograma.
    - renderer: dibujo de bboxes coloreados según estado.
"""
from src.compliance.verifier import (
    ComplianceStatus,
    Detection,
    PersonVerdict,
    FrameReport,
    verify_frame,
)

__all__ = [
    "ComplianceStatus",
    "Detection",
    "PersonVerdict",
    "FrameReport",
    "verify_frame",
]
