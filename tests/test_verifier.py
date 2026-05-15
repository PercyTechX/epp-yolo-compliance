"""Tests unitarios del verificador de cumplimiento de EPP.

Se construyen escenarios sintéticos a mano (sin imágenes reales). Cada
caso define una lista de `Detection` y asegura el `FrameReport` esperado.
Mínimo 10 casos según fase 4 del plan.
"""
from __future__ import annotations

import pytest

from src.compliance.verifier import (
    CLASS_HEAD_WITH_HELMET,
    CLASS_HEAD_WITHOUT_HELMET,
    CLASS_NO_VEST_PERSON,
    CLASS_PERSON,
    CLASS_VEST,
    ComplianceStatus,
    Detection,
    VerifierConfig,
    verify_frame,
)


# Cajas reutilizables para legibilidad de los casos.
PERSON_A = (100.0, 50.0, 200.0, 350.0)         # persona izquierda
PERSON_B = (400.0, 50.0, 500.0, 350.0)         # persona derecha
HEAD_HELMET_A = (120.0, 60.0, 180.0, 100.0)    # casco en tercio superior de A
HEAD_NO_HELMET_A = (120.0, 60.0, 180.0, 100.0) # misma posición, sin casco
VEST_A = (110.0, 150.0, 190.0, 250.0)          # chaleco dentro de A
HEAD_BELOW = (120.0, 280.0, 180.0, 320.0)      # cabeza fuera del tercio superior
VEST_OUT = (300.0, 150.0, 380.0, 250.0)        # chaleco fuera de cualquier persona


def _person(box=PERSON_A) -> Detection:
    return Detection(class_id=CLASS_PERSON, box=box, score=0.9)


def _head_helmet(box=HEAD_HELMET_A) -> Detection:
    return Detection(class_id=CLASS_HEAD_WITH_HELMET, box=box, score=0.9)


def _head_no_helmet(box=HEAD_NO_HELMET_A) -> Detection:
    return Detection(class_id=CLASS_HEAD_WITHOUT_HELMET, box=box, score=0.9)


def _vest(box=VEST_A) -> Detection:
    return Detection(class_id=CLASS_VEST, box=box, score=0.9)


def _no_vest_person(box=PERSON_B) -> Detection:
    return Detection(class_id=CLASS_NO_VEST_PERSON, box=box, score=0.9)


# ---------------------------------------------------------------------------
# Casos.
# ---------------------------------------------------------------------------


def test_caso_01_conforme_completo():
    """Persona con cabeza_con_casco y chaleco correctamente asociados."""
    dets = [_person(), _head_helmet(), _vest()]
    report = verify_frame(dets)
    assert len(report.verdicts) == 1
    assert report.verdicts[0].status == ComplianceStatus.CONFORME
    assert report.compliance_rate == pytest.approx(1.0)


def test_caso_02_no_conforme_sin_casco():
    """Persona con cabeza sin casco y chaleco -> NO_CONFORME (casco_ausente)."""
    dets = [_person(), _head_no_helmet(), _vest()]
    report = verify_frame(dets)
    v = report.verdicts[0]
    assert v.status == ComplianceStatus.NO_CONFORME
    assert "casco_ausente" in v.reasons
    assert report.compliance_rate == pytest.approx(0.0)


def test_caso_03_no_conforme_sin_chaleco():
    """Persona con casco pero sin chaleco -> NO_CONFORME (chaleco_ausente)."""
    dets = [_person(), _head_helmet()]
    report = verify_frame(dets)
    v = report.verdicts[0]
    assert v.status == ComplianceStatus.NO_CONFORME
    assert "chaleco_ausente" in v.reasons


def test_caso_04_no_conforme_doble_falla():
    """Persona sin casco y sin chaleco -> NO_CONFORME con dos razones."""
    dets = [_person(), _head_no_helmet()]
    report = verify_frame(dets)
    v = report.verdicts[0]
    assert v.status == ComplianceStatus.NO_CONFORME
    assert "casco_ausente" in v.reasons
    assert "chaleco_ausente" in v.reasons


def test_caso_05_indeterminado_sin_evidencia():
    """Persona sin cabeza ni chaleco que asociar -> INDETERMINADO."""
    dets = [_person()]
    report = verify_frame(dets)
    v = report.verdicts[0]
    assert v.status == ComplianceStatus.INDETERMINADO
    assert "sin_evidencia_asociada" in v.reasons
    # La tasa de cumplimiento ignora INDETERMINADOS: 0 sobre 0 -> 0.
    assert report.compliance_rate == 0.0


def test_caso_06_cabeza_fuera_de_tercio_superior_no_asocia():
    """Cabeza en la parte baja del bbox no debe asociarse a la persona."""
    head_low = Detection(class_id=CLASS_HEAD_WITH_HELMET, box=HEAD_BELOW, score=0.9)
    dets = [_person(), head_low, _vest()]
    report = verify_frame(dets)
    v = report.verdicts[0]
    # Como no se asoció cabeza, falla casco; chaleco sí asociado -> NO_CONFORME.
    assert v.status == ComplianceStatus.NO_CONFORME
    assert "cabeza_no_asociada" in v.reasons
    assert v.head_box is None


def test_caso_07_cabeza_sin_overlap_horizontal_no_asocia():
    """Cabeza muy desplazada en X (sin overlap) no debe asociarse."""
    head_far = Detection(
        class_id=CLASS_HEAD_WITH_HELMET, box=(500.0, 60.0, 540.0, 100.0), score=0.9
    )
    dets = [_person(), head_far, _vest()]
    report = verify_frame(dets)
    v = report.verdicts[0]
    assert v.head_box is None
    assert v.status == ComplianceStatus.NO_CONFORME


def test_caso_08_chaleco_fuera_no_asocia():
    """Chaleco no contenido en la persona no debe asociarse -> NO_CONFORME."""
    vest_far = Detection(class_id=CLASS_VEST, box=VEST_OUT, score=0.9)
    dets = [_person(), _head_helmet(), vest_far]
    report = verify_frame(dets)
    v = report.verdicts[0]
    assert v.status == ComplianceStatus.NO_CONFORME
    assert "chaleco_ausente" in v.reasons


def test_caso_09_no_vest_person_aislado_es_no_conforme():
    """Una persona detectada como `no_vest_person` sin `person` solapada debe reportarse."""
    dets = [_no_vest_person()]
    report = verify_frame(dets)
    assert len(report.verdicts) == 1
    v = report.verdicts[0]
    assert v.status == ComplianceStatus.NO_CONFORME
    assert "chaleco_ausente" in v.reasons


def test_caso_10_no_vest_person_redundante_no_duplica():
    """Si `no_vest_person` solapa con `person` ya procesada, no se duplica el veredicto."""
    # Cabeza con casco + chaleco para que `person` salga CONFORME; pero también
    # llega `no_vest_person` con casi el mismo bbox -> debe ignorarse.
    nv_overlapping = Detection(class_id=CLASS_NO_VEST_PERSON, box=PERSON_A, score=0.9)
    dets = [_person(), _head_helmet(), _vest(), nv_overlapping]
    report = verify_frame(dets)
    assert len(report.verdicts) == 1
    assert report.verdicts[0].status == ComplianceStatus.CONFORME


def test_caso_11_multiples_personas_tasa_correcta():
    """Dos personas: una conforme, una no -> tasa = 0.5."""
    person_b = _person(box=PERSON_B)
    head_b = Detection(
        class_id=CLASS_HEAD_WITHOUT_HELMET,
        box=(420.0, 60.0, 480.0, 100.0),
        score=0.9,
    )
    dets = [_person(), _head_helmet(), _vest(), person_b, head_b]
    report = verify_frame(dets)
    statuses = {v.status for v in report.verdicts}
    assert ComplianceStatus.CONFORME in statuses
    assert ComplianceStatus.NO_CONFORME in statuses
    assert report.compliance_rate == pytest.approx(0.5)


def test_caso_12_sin_detecciones_reporte_vacio():
    """Frame sin detecciones: reporte vacío y tasa neutra 0."""
    report = verify_frame([])
    assert report.verdicts == []
    assert report.compliance_rate == 0.0


def test_caso_13_umbral_contencion_configurable():
    """Subir el umbral de contención puede rechazar un chaleco apenas dentro."""
    # Chaleco parcialmente fuera: ~50% dentro.
    partial_vest = Detection(class_id=CLASS_VEST, box=(150.0, 150.0, 250.0, 250.0), score=0.9)
    dets = [_person(), _head_helmet(), partial_vest]

    cfg_laxo = VerifierConfig(min_containment_vest=0.4)
    cfg_estricto = VerifierConfig(min_containment_vest=0.9)

    r_laxo = verify_frame(dets, cfg_laxo)
    r_estricto = verify_frame(dets, cfg_estricto)

    assert r_laxo.verdicts[0].status == ComplianceStatus.CONFORME
    assert r_estricto.verdicts[0].status == ComplianceStatus.NO_CONFORME
    assert "chaleco_ausente" in r_estricto.verdicts[0].reasons


def test_caso_14_competencia_de_cabezas_elige_mayor_overlap():
    """Si hay dos cabezas candidatas, gana la que solapa más horizontalmente."""
    head_centered = Detection(
        class_id=CLASS_HEAD_WITH_HELMET, box=(120.0, 60.0, 180.0, 100.0), score=0.9
    )  # centrada en PERSON_A
    head_edge = Detection(
        class_id=CLASS_HEAD_WITHOUT_HELMET,
        box=(180.0, 60.0, 220.0, 100.0),
        score=0.9,
    )  # apenas tocando el borde derecho
    dets = [_person(), head_centered, head_edge, _vest()]
    report = verify_frame(dets)
    v = report.verdicts[0]
    # La cabeza centrada (con casco) debe ganar -> CONFORME.
    assert v.status == ComplianceStatus.CONFORME
    assert v.head_class == CLASS_HEAD_WITH_HELMET
