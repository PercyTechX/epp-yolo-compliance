"""Lógica determinista de verificación de cumplimiento de EPP.

Consume las detecciones de un fotograma (lista de `Detection`) y produce
un `FrameReport` con el veredicto por trabajador y la tasa de cumplimiento
global del fotograma. No usa información temporal entre fotogramas (por
decisión explícita del plan, fase 4).

**Estrategia de asociación (cabeza-ancla):**

En los datasets disponibles, la clase `person` es escasa (~2% de las
instancias) mientras que las cabezas con/sin casco son abundantes (≥95% mAP).
Por eso el verificador **se ancla en la CABEZA**, no en `person`:

1. Para cada cabeza detectada (con o sin casco), busca un `vest` debajo
   de la cabeza con alineación horizontal razonable (la región torso).
2. Asigna un estado según presencia de casco y chaleco asociados.
3. Si existe una detección `person` que contiene a la cabeza, se usa para
   el bbox de renderizado; si no, se construye un bbox virtual combinando
   cabeza + chaleco.

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
    containment,
)

# Ids de la taxonomía unificada. Cualquier cambio aquí debe replicarse en
# configs/data.yaml y configs/class_mapping.yaml.
CLASS_HEAD_WITH_HELMET = 0
CLASS_HEAD_WITHOUT_HELMET = 1
CLASS_PERSON = 2
CLASS_VEST = 3
CLASS_NO_VEST_PERSON = 4


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
    vest_box: Optional[BBox] = None
    reasons: List[str] = field(default_factory=list)


@dataclass
class FrameReport:
    """Reporte agregado de un fotograma."""

    verdicts: List[PersonVerdict]
    compliance_rate: float


@dataclass(frozen=True)
class VerifierConfig:
    """Umbrales ajustables del verificador.

    Expuestos como parámetros para que el operador pueda calibrar sin
    tocar la lógica.
    """

    # Factor del alto de la cabeza usado como margen vertical máximo
    # para asociar un chaleco. Si vest.top > head.bottom + max_vest_drop * head_h
    # el chaleco se considera demasiado lejos.
    max_vest_drop_in_head_heights: float = 5.0

    # Margen horizontal (en fracción del ancho de la cabeza) que permite
    # que el centro horizontal de la cabeza caiga fuera del rango del chaleco
    # antes de descartarlo (tolera diagonal moderada del torso).
    horizontal_margin_in_head_widths: float = 0.5

    # Score mínimo de detección para considerar evidencia.
    min_detection_score: float = 0.0


# ---------------------------------------------------------------------------
# Helpers de asociación cabeza-ancla.
# ---------------------------------------------------------------------------


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
        if det.score < 0:  # placeholder, no se usa salvo que cfg.min_score>0
            continue
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


def _find_vest_below_head(
    head: Detection,
    vests: List[Detection],
    used: set,
    cfg: VerifierConfig,
) -> Tuple[Optional[int], Optional[Detection]]:
    """Encuentra el chaleco más cercano debajo de la cabeza con alineación válida.

    Reglas:
      1. El centro vertical del chaleco está por debajo del centro vertical
         de la cabeza.
      2. El centro horizontal de la cabeza cae dentro del rango horizontal
         del chaleco (con un margen igual a media anchura de cabeza).
      3. El tope del chaleco está como máximo a `max_vest_drop_in_head_heights`
         alturas de cabeza por debajo del fondo de la cabeza.
    Entre los candidatos válidos se elige el de menor distancia vertical
    (más cercano debajo). Los índices ya usados no se reasignan.
    """
    hx1, hy1, hx2, hy2 = head.box
    head_w = max(0.0, hx2 - hx1)
    head_h = max(0.0, hy2 - hy1)
    head_center_x = (hx1 + hx2) / 2.0
    head_center_y = (hy1 + hy2) / 2.0

    best_idx: Optional[int] = None
    best_dist = float("inf")

    margin = cfg.horizontal_margin_in_head_widths * head_w
    max_drop = cfg.max_vest_drop_in_head_heights * head_h

    for i, vest in enumerate(vests):
        if i in used:
            continue
        vx1, vy1, vx2, vy2 = vest.box
        vest_center_y = (vy1 + vy2) / 2.0
        # Regla 1: chaleco debajo de la cabeza.
        if vest_center_y <= head_center_y:
            continue
        # Regla 2: alineación horizontal con margen.
        if not (vx1 - margin <= head_center_x <= vx2 + margin):
            continue
        # Regla 3: cercanía vertical.
        drop = max(0.0, vy1 - hy2)
        if drop > max_drop:
            continue
        if drop < best_dist:
            best_dist = drop
            best_idx = i

    if best_idx is None:
        return None, None
    return best_idx, vests[best_idx]


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


def _virtual_worker_bbox(head: Detection, vest: Optional[Detection]) -> BBox:
    """Construye un bbox virtual para el trabajador cuando no hay `person` detectado.

    Si hay chaleco asociado: une cabeza + chaleco. Si no: extiende la cabeza
    hacia abajo varias alturas para sugerir el cuerpo aproximado.
    """
    hx1, hy1, hx2, hy2 = head.box
    if vest is None:
        head_h = max(1.0, hy2 - hy1)
        return (hx1, hy1, hx2, hy2 + 3.0 * head_h)
    vx1, vy1, vx2, vy2 = vest.box
    return (min(hx1, vx1), min(hy1, vy1), max(hx2, vx2), max(hy2, vy2))


def _is_already_covered_by_heads(
    box: BBox, heads: List[Detection], min_head_containment: float = 0.5
) -> bool:
    """Indica si alguna cabeza está mayormente contenida dentro del `box`.

    Usa contención (no IoU) porque una cabeza pequeña dentro de un bbox
    grande de persona típicamente da IoU bajo pero contención cercana a 1.
    Si una cabeza ya está cubierta por ese bbox, el bbox refiere al mismo
    trabajador y no necesita un veredicto duplicado.
    """
    for h in heads:
        if containment(h.box, box) >= min_head_containment:
            return True
    return False


# ---------------------------------------------------------------------------
# Función pública principal.
# ---------------------------------------------------------------------------


def verify_frame(
    detections: List[Detection],
    cfg: Optional[VerifierConfig] = None,
) -> FrameReport:
    """Aplica las reglas de cumplimiento sobre las detecciones de un fotograma.

    Estrategia cabeza-ancla:
      - Para cada cabeza (con o sin casco), busca chaleco debajo.
      - head_with_helmet + vest        → CONFORME
      - head_with_helmet sin vest      → NO_CONFORME ("chaleco_no_visible")
      - head_without_helmet            → NO_CONFORME ("casco_ausente",
                                          + "chaleco_no_visible" si no hay vest)
      - vest sin cabeza asociada       → INDETERMINADO ("cabeza_no_visible")
      - no_vest_person sin solapamiento con cabezas → NO_CONFORME ("chaleco_ausente")
    """
    cfg = cfg or VerifierConfig()
    heads_helmet, heads_no_helmet, persons, vests, no_vests = _split_by_class(
        detections
    )

    # Combinamos cabezas en una sola lista, marcando su clase.
    all_heads: List[Tuple[Detection, int]] = [
        (h, CLASS_HEAD_WITH_HELMET) for h in heads_helmet
    ] + [
        (h, CLASS_HEAD_WITHOUT_HELMET) for h in heads_no_helmet
    ]

    verdicts: List[PersonVerdict] = []
    used_vest_indices: set = set()

    for head, head_class in all_heads:
        if head.score < cfg.min_detection_score:
            continue

        vest_idx, vest_det = _find_vest_below_head(
            head, vests, used_vest_indices, cfg
        )
        if vest_idx is not None:
            used_vest_indices.add(vest_idx)

        person_det = _find_person_for_head(head, persons)

        helmet_ok = head_class == CLASS_HEAD_WITH_HELMET
        vest_ok = vest_det is not None
        reasons: List[str] = []

        if helmet_ok and vest_ok:
            status = ComplianceStatus.CONFORME
        elif helmet_ok and not vest_ok:
            status = ComplianceStatus.NO_CONFORME
            reasons.append("chaleco_no_visible")
        elif not helmet_ok and vest_ok:
            status = ComplianceStatus.NO_CONFORME
            reasons.append("casco_ausente")
        else:
            status = ComplianceStatus.NO_CONFORME
            reasons.append("casco_ausente")
            reasons.append("chaleco_no_visible")

        bbox_for_render = (
            person_det.box if person_det else _virtual_worker_bbox(head, vest_det)
        )

        verdicts.append(
            PersonVerdict(
                person_box=bbox_for_render,
                status=status,
                head_class=head_class,
                head_box=head.box,
                vest_box=vest_det.box if vest_det else None,
                reasons=reasons,
            )
        )

    # Chalecos sin cabeza asociada → INDETERMINADO.
    for i, vest in enumerate(vests):
        if i in used_vest_indices:
            continue
        verdicts.append(
            PersonVerdict(
                person_box=vest.box,
                status=ComplianceStatus.INDETERMINADO,
                vest_box=vest.box,
                reasons=["cabeza_no_visible"],
            )
        )

    # no_vest_person que no solapen con cabezas ya procesadas → NO_CONFORME.
    head_boxes = [h for h, _ in all_heads]
    for nv in no_vests:
        if _is_already_covered_by_heads(nv.box, head_boxes):
            continue
        verdicts.append(
            PersonVerdict(
                person_box=nv.box,
                status=ComplianceStatus.NO_CONFORME,
                reasons=["chaleco_ausente"],
            )
        )

    determinate = [
        v for v in verdicts if v.status != ComplianceStatus.INDETERMINADO
    ]
    conformes = [v for v in determinate if v.status == ComplianceStatus.CONFORME]
    rate = (len(conformes) / len(determinate)) if determinate else 0.0
    return FrameReport(verdicts=verdicts, compliance_rate=rate)
