"""Tests unitarios del verificador de cumplimiento (regla solo-casco).

Se construyen escenarios sintéticos a mano (sin imágenes reales). Cada
caso define una lista de `Detection` y asegura el `FrameReport` esperado.
Mínimo 10 casos según fase 4 del plan.

Convención de cajas en estos tests (xyxy):
    Cabeza:  ~60×40 px arriba del trabajador.
    Persona: ~100×300 px que envuelve al trabajador completo.
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


# Cajas estándar de un trabajador A.
HEAD_A = (120.0, 60.0, 180.0, 100.0)       # cabeza arriba, 60×40
PERSON_A = (100.0, 50.0, 200.0, 350.0)     # persona completa
VEST_A = (110.0, 150.0, 190.0, 250.0)      # chaleco (ya no usado por el verificador)

# Cajas de un trabajador B (a la derecha).
HEAD_B = (420.0, 60.0, 480.0, 100.0)
PERSON_B = (400.0, 50.0, 500.0, 350.0)

# Persona aislada (sin cabeza detectable adentro).
PERSON_LONELY = (700.0, 50.0, 800.0, 350.0)


def _head_helmet(box=HEAD_A) -> Detection:
    return Detection(class_id=CLASS_HEAD_WITH_HELMET, box=box, score=0.9)


def _head_no_helmet(box=HEAD_A) -> Detection:
    return Detection(class_id=CLASS_HEAD_WITHOUT_HELMET, box=box, score=0.9)


def _vest(box=VEST_A) -> Detection:
    return Detection(class_id=CLASS_VEST, box=box, score=0.9)


def _person(box=PERSON_A) -> Detection:
    return Detection(class_id=CLASS_PERSON, box=box, score=0.9)


def _no_vest_person(box=PERSON_B) -> Detection:
    return Detection(class_id=CLASS_NO_VEST_PERSON, box=box, score=0.9)


# ---------------------------------------------------------------------------
# Casos.
# ---------------------------------------------------------------------------


def test_caso_01_conforme_cabeza_con_casco_aislada():
    """Cabeza con casco sin más detecciones -> CONFORME."""
    report = verify_frame([_head_helmet()])
    assert len(report.verdicts) == 1
    v = report.verdicts[0]
    assert v.status == ComplianceStatus.CONFORME
    assert v.head_class == CLASS_HEAD_WITH_HELMET
    assert v.head_box == HEAD_A
    assert v.vest_box is None
    assert report.compliance_rate == pytest.approx(1.0)


def test_caso_02_no_conforme_cabeza_sin_casco_aislada():
    """Cabeza sin casco sin más detecciones -> NO_CONFORME por casco_ausente."""
    report = verify_frame([_head_no_helmet()])
    v = report.verdicts[0]
    assert v.status == ComplianceStatus.NO_CONFORME
    assert "casco_ausente" in v.reasons
    assert report.compliance_rate == pytest.approx(0.0)


def test_caso_03_conforme_con_person_para_render():
    """Cabeza con casco + persona que la contiene -> CONFORME (bbox = persona)."""
    report = verify_frame([_head_helmet(), _person()])
    v = report.verdicts[0]
    assert v.status == ComplianceStatus.CONFORME
    assert v.person_box == PERSON_A  # se usa el bbox de la persona, no el virtual


def test_caso_04_no_conforme_con_person_para_render():
    """Cabeza sin casco + persona que la contiene -> NO_CONFORME, bbox = persona."""
    report = verify_frame([_head_no_helmet(), _person()])
    v = report.verdicts[0]
    assert v.status == ComplianceStatus.NO_CONFORME
    assert "casco_ausente" in v.reasons
    assert v.person_box == PERSON_A


def test_caso_05_chaleco_se_ignora_silenciosamente():
    """Detecciones de chaleco NO generan veredictos en la regla solo-casco."""
    report = verify_frame([_vest()])
    assert report.verdicts == []
    assert report.compliance_rate == 0.0


def test_caso_06_chaleco_no_afecta_veredicto_de_cabeza():
    """Una cabeza con casco + un chaleco al lado: solo cuenta la cabeza."""
    report = verify_frame([_head_helmet(), _vest()])
    assert len(report.verdicts) == 1
    v = report.verdicts[0]
    assert v.status == ComplianceStatus.CONFORME
    # vest_box debe ser None: el verificador no la usa.
    assert v.vest_box is None


def test_caso_07_no_vest_person_se_ignora():
    """Detecciones de no_vest_person NO generan veredictos en la regla solo-casco."""
    report = verify_frame([_no_vest_person()])
    assert report.verdicts == []


def test_caso_08_persona_sin_cabeza_visible_es_indeterminado():
    """Persona detectada sin ninguna cabeza adentro -> INDETERMINADO."""
    report = verify_frame([_person(box=PERSON_LONELY)])
    assert len(report.verdicts) == 1
    v = report.verdicts[0]
    assert v.status == ComplianceStatus.INDETERMINADO
    assert "cabeza_no_visible" in v.reasons


def test_caso_09_persona_con_cabeza_adentro_no_duplica_veredicto():
    """Si la cabeza ya generó veredicto, la persona que la contiene NO genera otro."""
    report = verify_frame([_head_helmet(), _person()])
    assert len(report.verdicts) == 1
    assert report.verdicts[0].status == ComplianceStatus.CONFORME


def test_caso_10_sin_detecciones_reporte_vacio():
    """Frame sin detecciones: reporte vacío y tasa neutra."""
    report = verify_frame([])
    assert report.verdicts == []
    assert report.compliance_rate == 0.0


def test_caso_11_dos_trabajadores_uno_con_casco_uno_sin():
    """Dos trabajadores: uno con casco (CONFORME), uno sin (NO_CONFORME). Tasa 0.5."""
    dets = [
        _head_helmet(HEAD_A),
        _head_no_helmet(HEAD_B),
    ]
    report = verify_frame(dets)
    assert len(report.verdicts) == 2
    statuses = {v.status for v in report.verdicts}
    assert ComplianceStatus.CONFORME in statuses
    assert ComplianceStatus.NO_CONFORME in statuses
    assert report.compliance_rate == pytest.approx(0.5)


def test_caso_12_dos_cabezas_conformes_tasa_100():
    """Dos cabezas con casco -> 2 CONFORME, tasa 100%."""
    report = verify_frame([_head_helmet(HEAD_A), _head_helmet(HEAD_B)])
    assert len(report.verdicts) == 2
    assert all(v.status == ComplianceStatus.CONFORME for v in report.verdicts)
    assert report.compliance_rate == pytest.approx(1.0)


def test_caso_13_indeterminado_no_cuenta_en_tasa():
    """Una cabeza con casco + una persona sin cabeza: tasa 100% (solo el CONFORME cuenta)."""
    report = verify_frame([_head_helmet(), _person(box=PERSON_LONELY)])
    assert len(report.verdicts) == 2
    statuses = {v.status for v in report.verdicts}
    assert ComplianceStatus.CONFORME in statuses
    assert ComplianceStatus.INDETERMINADO in statuses
    # Tasa: 1 CONFORME / 1 determinado = 1.0
    assert report.compliance_rate == pytest.approx(1.0)


def test_caso_14_escenario_mixto_complejo():
    """Escena con varios tipos de detecciones: solo cabezas + personas cuentan."""
    dets = [
        _head_helmet(HEAD_A),                              # CONFORME (asociado a PERSON_A)
        _head_no_helmet(HEAD_B),                           # NO_CONFORME
        _person(PERSON_A),                                 # absorbido por HEAD_A
        _person(PERSON_B),                                 # absorbido por HEAD_B (sin casco)
        _person(PERSON_LONELY),                            # INDETERMINADO
        _vest(),                                           # IGNORADO
        _no_vest_person(),                                 # IGNORADO
    ]
    report = verify_frame(dets)
    # Esperado: 3 veredictos (2 por cabezas + 1 por persona huérfana).
    assert len(report.verdicts) == 3
    statuses = [v.status for v in report.verdicts]
    assert statuses.count(ComplianceStatus.CONFORME) == 1
    assert statuses.count(ComplianceStatus.NO_CONFORME) == 1
    assert statuses.count(ComplianceStatus.INDETERMINADO) == 1
    # Tasa: 1 CONFORME / 2 determinados = 0.5
    assert report.compliance_rate == pytest.approx(0.5)


def test_caso_15_umbral_de_score_descarta_detecciones_debiles():
    """Cabezas con score muy bajo se descartan si cfg lo exige."""
    weak_head = Detection(
        class_id=CLASS_HEAD_WITH_HELMET, box=HEAD_A, score=0.05
    )
    cfg = VerifierConfig(min_detection_score=0.5)
    report = verify_frame([weak_head], cfg)
    assert report.verdicts == []
