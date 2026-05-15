"""Lógica determinista de verificación de cumplimiento de EPP.

Consume las detecciones de un fotograma (lista de `Detection`) y produce
un `FrameReport` con el veredicto por persona y la tasa de cumplimiento
global del fotograma. No usa información temporal entre fotogramas (por
decisión explícita del plan, fase 4).

Convenciones de clases (deben coincidir con configs/data.yaml):
    0: head_with_helmet
    1: head_without_helmet
    2: person
    3: vest
    4: no_vest_person
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from src.compliance.geometry import (
    BBox,
    associate_head_to_person,
    associate_vest_to_person,
    containment,
    horizontal_overlap_ratio,
)

# Ids de la taxonomía unificada. Cualquier cambio aquí debe replicarse en
# configs/data.yaml y configs/class_mapping.yaml.
CLASS_HEAD_WITH_HELMET = 0
CLASS_HEAD_WITHOUT_HELMET = 1
CLASS_PERSON = 2
CLASS_VEST = 3
CLASS_NO_VEST_PERSON = 4


class ComplianceStatus(str, Enum):
    """Estado de cumplimiento por persona detectada."""

    CONFORME = "CONFORME"
    NO_CONFORME = "NO_CONFORME"
    INDETERMINADO = "INDETERMINADO"


@dataclass(frozen=True)
class Detection:
    """Detección individual emitida por YOLO.

    `box` en formato xyxy absoluto (píxeles). `score` es la confianza
    del detector. `class_id` debe pertenecer a la taxonomía unificada.
    """

    class_id: int
    box: BBox
    score: float = 1.0


@dataclass
class PersonVerdict:
    """Resultado de la verificación para una persona detectada."""

    person_box: BBox
    status: ComplianceStatus
    head_class: Optional[int] = None        # id de la cabeza asociada, si la hay
    head_box: Optional[BBox] = None
    vest_box: Optional[BBox] = None
    reasons: List[str] = field(default_factory=list)


@dataclass
class FrameReport:
    """Reporte agregado de un fotograma."""

    verdicts: List[PersonVerdict]
    compliance_rate: float  # fracción CONFORMES sobre personas con veredicto definido


@dataclass(frozen=True)
class VerifierConfig:
    """Umbrales ajustables del verificador.

    Expuestos como parámetros para que el operador pueda calibrar sin
    tocar la lógica. Coherentes con el plan: simplicidad y auditabilidad.
    """

    min_horizontal_overlap_head: float = 0.5
    min_containment_vest: float = 0.6
    # Score mínimo de detección para considerar evidencia. Por defecto 0
    # para no filtrar nada aquí (Ultralytics ya filtró por conf en NMS).
    min_detection_score: float = 0.0


def _split_by_class(detections: List[Detection]) -> Tuple[
    List[Detection], List[Detection], List[Detection], List[Detection], List[Detection]
]:
    """Separa las detecciones por id de clase."""
    heads_helmet: List[Detection] = []
    heads_no_helmet: List[Detection] = []
    persons: List[Detection] = []
    vests: List[Detection] = []
    no_vests: List[Detection] = []
    for det in detections:
        if det.class_id == CLASS_HEAD_WITH_HELMET:
            heads_helmet.append(det)
        elif det.class_id == CLASS_HEAD_WITHOUT_HELMET:
            heads_no_helmet.append(det)
        elif det.class_id == CLASS_PERSON:
            persons.append(det)
        elif det.class_id == CLASS_VEST:
            vests.append(det)
        elif det.class_id == CLASS_NO_VEST_PERSON:
            no_vests.append(det)
    return heads_helmet, heads_no_helmet, persons, vests, no_vests


def _find_associated_head(
    person: Detection,
    heads_helmet: List[Detection],
    heads_no_helmet: List[Detection],
    cfg: VerifierConfig,
) -> Tuple[Optional[Detection], Optional[int]]:
    """Encuentra la cabeza más probable asociada a una persona.

    Devuelve (detección_cabeza, class_id) o (None, None). Si compiten una
    cabeza con casco y una sin casco para la misma persona, gana la que
    tenga mayor solapamiento horizontal (criterio simple, determinista).
    """
    best: Optional[Detection] = None
    best_class: Optional[int] = None
    best_overlap = -1.0

    for head in heads_helmet + heads_no_helmet:
        if not associate_head_to_person(
            head.box, person.box, cfg.min_horizontal_overlap_head
        ):
            continue
        overlap = horizontal_overlap_ratio(head.box, person.box)
        if overlap > best_overlap:
            best_overlap = overlap
            best = head
            best_class = head.class_id
    return best, best_class


def _find_associated_vest(
    person: Detection,
    vests: List[Detection],
    cfg: VerifierConfig,
) -> Optional[Detection]:
    """Encuentra el chaleco más probable asociado a una persona.

    Selecciona el de mayor contención dentro del bbox de la persona.
    """
    best: Optional[Detection] = None
    best_score = -1.0
    for vest in vests:
        if not associate_vest_to_person(
            vest.box, person.box, cfg.min_containment_vest
        ):
            continue
        c = containment(vest.box, person.box)
        if c > best_score:
            best_score = c
            best = vest
    return best


def verify_frame(
    detections: List[Detection],
    cfg: Optional[VerifierConfig] = None,
) -> FrameReport:
    """Aplica las reglas de cumplimiento sobre las detecciones de un fotograma.

    Reglas (resumen):
      - CONFORME      : la persona tiene cabeza_con_casco asociada Y chaleco asociado.
      - NO_CONFORME   : se asocia cabeza sin casco, O la persona aparece como
                        `no_vest_person`, O no se encuentra chaleco pero sí
                        cabeza con casco (chaleco faltante).
      - INDETERMINADO : no se puede asociar ni cabeza ni chaleco a la persona.

    Las personas que solo aparecen como `no_vest_person` (sin bbox `person`
    explícito) se reportan directamente como NO_CONFORME con razón
    "vest_ausente".
    """
    cfg = cfg or VerifierConfig()
    heads_helmet, heads_no_helmet, persons, vests, no_vests = _split_by_class(
        detections
    )

    verdicts: List[PersonVerdict] = []

    for person in persons:
        head_det, head_class = _find_associated_head(
            person, heads_helmet, heads_no_helmet, cfg
        )
        vest_det = _find_associated_vest(person, vests, cfg)

        reasons: List[str] = []
        status: ComplianceStatus

        helmet_ok = head_class == CLASS_HEAD_WITH_HELMET
        vest_ok = vest_det is not None

        if head_det is None and vest_det is None:
            status = ComplianceStatus.INDETERMINADO
            reasons.append("sin_evidencia_asociada")
        elif helmet_ok and vest_ok:
            status = ComplianceStatus.CONFORME
        else:
            status = ComplianceStatus.NO_CONFORME
            if head_class == CLASS_HEAD_WITHOUT_HELMET:
                reasons.append("casco_ausente")
            elif head_det is None:
                reasons.append("cabeza_no_asociada")
            if not vest_ok:
                reasons.append("chaleco_ausente")

        verdicts.append(
            PersonVerdict(
                person_box=person.box,
                status=status,
                head_class=head_class,
                head_box=head_det.box if head_det else None,
                vest_box=vest_det.box if vest_det else None,
                reasons=reasons,
            )
        )

    # Detecciones de `no_vest_person` que no se solapan con personas ya
    # procesadas se reportan como NO_CONFORME por chaleco ausente.
    for nv in no_vests:
        if _is_already_covered(nv.box, persons):
            continue
        verdicts.append(
            PersonVerdict(
                person_box=nv.box,
                status=ComplianceStatus.NO_CONFORME,
                reasons=["chaleco_ausente"],
            )
        )

    determinate = [v for v in verdicts if v.status != ComplianceStatus.INDETERMINADO]
    conformes = [v for v in determinate if v.status == ComplianceStatus.CONFORME]
    rate = (len(conformes) / len(determinate)) if determinate else 0.0
    return FrameReport(verdicts=verdicts, compliance_rate=rate)


def _is_already_covered(box: BBox, persons: List[Detection], iou_threshold: float = 0.5) -> bool:
    """Indica si una caja se solapa significativamente con alguna persona ya procesada.

    Evita contar doble cuando un mismo trabajador aparece etiquetado tanto
    como `person` como `no_vest_person`.
    """
    from src.compliance.geometry import iou as _iou

    for p in persons:
        if _iou(box, p.box) >= iou_threshold:
            return True
    return False
