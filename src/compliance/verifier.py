"""Lógica determinista de verificación de cumplimiento de casco.

Consume las detecciones de un fotograma (lista de `Detection`) y produce
un `FrameReport` con el veredicto por trabajador y la tasa de cumplimiento
global del fotograma. No usa información temporal entre fotogramas.

**Regla operativa: solo CASCO.**

La regla original del enunciado pedía verificar casco Y chaleco. Esta
versión se redefinió a "solo casco" por dos razones documentadas en el
informe (Sección 5):

  1. **Contexto local**: en obras pequeñas y medianas en Perú/Latam, el
     chaleco reflectivo no es uniforme universal. La prenda típica es
     polo manga larga + casco. El casco sí es obligatorio en todos los
     contextos.

  2. **Limitación de datos**: los datasets combinados (SHWD + Hard Hat
     + PPE) tienen anotación parcial no superpuesta, lo que hace que el
     modelo casi nunca firme cabeza + chaleco simultáneamente (3.3% de
     co-ocurrencia). La regla del chaleco quedaba operativamente
     bloqueada por este sesgo.

La nueva regla simplifica el verificador:
  - Cabeza CON casco detectada     -> CONFORME
  - Cabeza SIN casco detectada     -> NO_CONFORME ("casco_ausente")
  - Persona detectada sin cabeza   -> INDETERMINADO ("cabeza_no_visible")
  - Detecciones de chaleco / no_vest_person se IGNORAN.

Convenciones de clases (deben coincidir con configs/data.yaml):
    0: head_with_helmet
    1: head_without_helmet
    2: person
    3: vest               (se ignora en la regla actual)
    4: no_vest_person     (se ignora en la regla actual)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from src.compliance.geometry import (
    BBox,
    associate_head_to_person,
)

# Ids de la taxonomía unificada. Cualquier cambio aquí debe replicarse en
# configs/data.yaml y configs/class_mapping.yaml.
CLASS_HEAD_WITH_HELMET = 0
CLASS_HEAD_WITHOUT_HELMET = 1
CLASS_PERSON = 2
CLASS_VEST = 3            # ignorado en la regla solo-casco
CLASS_NO_VEST_PERSON = 4  # ignorado en la regla solo-casco


class ComplianceStatus(str, Enum):
    """Estado de cumplimiento por trabajador detectado."""

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
    """Resultado de la verificación para un trabajador detectado."""

    person_box: BBox                       # bbox usado para renderizar el veredicto
    status: ComplianceStatus
    head_class: Optional[int] = None       # id de la cabeza asociada, si la hay
    head_box: Optional[BBox] = None
    head_score: Optional[float] = None     # confianza del detector para la cabeza
    vest_box: Optional[BBox] = None        # se mantiene en la dataclass por compatibilidad
    reasons: List[str] = field(default_factory=list)


@dataclass
class FrameReport:
    """Reporte agregado de un fotograma."""

    verdicts: List[PersonVerdict]
    compliance_rate: float


@dataclass(frozen=True)
class VerifierConfig:
    """Umbrales ajustables del verificador.

    La regla solo-casco tiene muy pocos parámetros porque la decisión
    es directa una vez detectada la cabeza.
    """

    # Score mínimo de detección para considerar evidencia.
    min_detection_score: float = 0.0


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _split_by_class(detections: List[Detection]) -> Tuple[
    List[Detection], List[Detection], List[Detection]
]:
    """Separa las detecciones por id de clase.

    Solo retorna las clases relevantes para la regla solo-casco:
    cabezas (con/sin casco) y personas. Las clases `vest` y
    `no_vest_person` se descartan silenciosamente.
    """
    heads_helmet: List[Detection] = []
    heads_no_helmet: List[Detection] = []
    persons: List[Detection] = []
    for det in detections:
        if det.class_id == CLASS_HEAD_WITH_HELMET:
            heads_helmet.append(det)
        elif det.class_id == CLASS_HEAD_WITHOUT_HELMET:
            heads_no_helmet.append(det)
        elif det.class_id == CLASS_PERSON:
            persons.append(det)
        # vest y no_vest_person: ignorados en la regla actual.
    return heads_helmet, heads_no_helmet, persons


def _find_person_for_head(
    head: Detection, persons: List[Detection]
) -> Optional[Detection]:
    """Encuentra una detección de `person` cuya cabeza coincida con la cabeza dada.

    Usa la heurística clásica (cabeza en tercio superior + solapamiento
    horizontal). Útil solo para enriquecer el render con un bbox grande;
    no afecta el veredicto.
    """
    for person in persons:
        if associate_head_to_person(head.box, person.box):
            return person
    return None


def _head_inside_person(head: Detection, person: Detection) -> bool:
    """Indica si una cabeza pertenece al bbox de una persona."""
    return associate_head_to_person(head.box, person.box)


def _virtual_worker_bbox(head: Detection) -> BBox:
    """Construye un bbox virtual para el trabajador cuando no hay `person` detectado.

    Extiende la cabeza hacia abajo varias alturas para sugerir el cuerpo
    aproximado. Útil para el render cuando solo se detectó la cabeza.
    """
    hx1, hy1, hx2, hy2 = head.box
    head_h = max(1.0, hy2 - hy1)
    return (hx1, hy1, hx2, hy2 + 3.0 * head_h)


# ---------------------------------------------------------------------------
# Función pública principal.
# ---------------------------------------------------------------------------


def verify_frame(
    detections: List[Detection],
    cfg: Optional[VerifierConfig] = None,
) -> FrameReport:
    """Aplica la regla solo-casco sobre las detecciones de un fotograma.

    Estrategia:
      - Para cada cabeza detectada (con o sin casco) se emite un veredicto
        inmediato:
            head_with_helmet     -> CONFORME
            head_without_helmet  -> NO_CONFORME ("casco_ausente")
      - Para cada `person` detectada que NO contiene una cabeza visible
        en su bbox, se emite INDETERMINADO ("cabeza_no_visible"): el
        sistema reconoce que ve un trabajador pero no puede juzgarlo
        sin ver su cabeza.
      - Las detecciones de `vest` y `no_vest_person` se descartan.
    """
    cfg = cfg or VerifierConfig()
    heads_helmet, heads_no_helmet, persons = _split_by_class(detections)

    # Lista combinada de cabezas con su clase.
    all_heads: List[Tuple[Detection, int]] = [
        (h, CLASS_HEAD_WITH_HELMET) for h in heads_helmet
    ] + [
        (h, CLASS_HEAD_WITHOUT_HELMET) for h in heads_no_helmet
    ]

    verdicts: List[PersonVerdict] = []

    # 1) Veredicto por cada cabeza detectada.
    for head, head_class in all_heads:
        if head.score < cfg.min_detection_score:
            continue

        # Buscar persona que contenga la cabeza, para enriquecer el render.
        person_det = _find_person_for_head(head, persons)

        if head_class == CLASS_HEAD_WITH_HELMET:
            status = ComplianceStatus.CONFORME
            reasons: List[str] = []
        else:
            status = ComplianceStatus.NO_CONFORME
            reasons = ["casco_ausente"]

        bbox_for_render = (
            person_det.box if person_det else _virtual_worker_bbox(head)
        )

        verdicts.append(
            PersonVerdict(
                person_box=bbox_for_render,
                status=status,
                head_class=head_class,
                head_box=head.box,
                head_score=head.score,
                vest_box=None,
                reasons=reasons,
            )
        )

    # 2) INDETERMINADO por cada persona detectada sin cabeza adentro.
    all_head_dets = [h for h, _ in all_heads]
    for person in persons:
        if person.score < cfg.min_detection_score:
            continue
        if any(_head_inside_person(h, person) for h in all_head_dets):
            # Ya cubierta por una cabeza procesada arriba.
            continue
        verdicts.append(
            PersonVerdict(
                person_box=person.box,
                status=ComplianceStatus.INDETERMINADO,
                reasons=["cabeza_no_visible"],
            )
        )

    # 3) Tasa de cumplimiento sobre veredictos definitivos.
    determinate = [
        v for v in verdicts if v.status != ComplianceStatus.INDETERMINADO
    ]
    conformes = [v for v in determinate if v.status == ComplianceStatus.CONFORME]
    rate = (len(conformes) / len(determinate)) if determinate else 0.0
    return FrameReport(verdicts=verdicts, compliance_rate=rate)
