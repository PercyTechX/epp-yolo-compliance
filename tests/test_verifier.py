"""Tests unitarios del verificador de cumplimiento de EPP (estrategia cabeza-ancla).

Se construyen escenarios sintéticos a mano (sin imágenes reales). Cada
caso define una lista de `Detection` y asegura el `FrameReport` esperado.
Mínimo 10 casos según fase 4 del plan.

Convención de cajas en estos tests (xyxy):
    Cabeza:  ~60×40 px arriba del trabajador.
    Chaleco: ~80×100 px en la zona del torso, debajo de la cabeza.
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
VEST_A = (110.0, 150.0, 190.0, 250.0)      # chaleco debajo (gap 50px ~ 1.25 alto cabeza)
PERSON_A = (100.0, 50.0, 200.0, 350.0)     # persona completa que envuelve cabeza y chaleco

# Cajas de un trabajador B (a la derecha).
HEAD_B = (420.0, 60.0, 480.0, 100.0)
VEST_B = (410.0, 150.0, 490.0, 250.0)
PERSON_B = (400.0, 50.0, 500.0, 350.0)

# Cajas patológicas.
VEST_FAR_BELOW = (110.0, 400.0, 190.0, 500.0)  # demasiado lejos debajo de HEAD_A
VEST_NOT_ALIGNED = (300.0, 150.0, 380.0, 250.0)  # horizontalmente desplazado
VEST_ABOVE_HEAD = (110.0, 0.0, 190.0, 30.0)    # arriba de la cabeza (imposible)


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


def test_caso_01_conforme_casco_y_chaleco_asociados():
    """Cabeza con casco + chaleco debajo correctamente asociados -> CONFORME."""
    report = verify_frame([_head_helmet(), _vest()])
    assert len(report.verdicts) == 1
    v = report.verdicts[0]
    assert v.status == ComplianceStatus.CONFORME
    assert v.head_class == CLASS_HEAD_WITH_HELMET
    assert v.head_box == HEAD_A
    assert v.vest_box == VEST_A
    assert report.compliance_rate == pytest.approx(1.0)


def test_caso_02_no_conforme_sin_casco_con_chaleco():
    """Cabeza sin casco + chaleco debajo -> NO_CONFORME por casco ausente."""
    report = verify_frame([_head_no_helmet(), _vest()])
    v = report.verdicts[0]
    assert v.status == ComplianceStatus.NO_CONFORME
    assert "casco_ausente" in v.reasons
    assert "chaleco_no_visible" not in v.reasons
    assert report.compliance_rate == pytest.approx(0.0)


def test_caso_03_no_conforme_con_casco_sin_chaleco():
    """Cabeza con casco sin chaleco -> NO_CONFORME por chaleco no visible."""
    report = verify_frame([_head_helmet()])
    v = report.verdicts[0]
    assert v.status == ComplianceStatus.NO_CONFORME
    assert "chaleco_no_visible" in v.reasons
    assert v.vest_box is None


def test_caso_04_no_conforme_doble_falla():
    """Cabeza sin casco sin chaleco -> NO_CONFORME con dos razones."""
    report = verify_frame([_head_no_helmet()])
    v = report.verdicts[0]
    assert v.status == ComplianceStatus.NO_CONFORME
    assert "casco_ausente" in v.reasons
    assert "chaleco_no_visible" in v.reasons


def test_caso_05_chaleco_demasiado_lejos_debajo_no_asocia():
    """Chaleco muy abajo de la cabeza no debe asociarse -> NO_CONFORME por chaleco."""
    vest_far = Detection(class_id=CLASS_VEST, box=VEST_FAR_BELOW, score=0.9)
    report = verify_frame([_head_helmet(), vest_far])
    # Hay dos veredictos: el de la cabeza (sin chaleco asociado) y el del
    # chaleco huérfano (INDETERMINADO).
    statuses = {v.status for v in report.verdicts}
    assert ComplianceStatus.NO_CONFORME in statuses   # cabeza sin chaleco
    assert ComplianceStatus.INDETERMINADO in statuses # chaleco sin cabeza


def test_caso_06_chaleco_desalineado_horizontal_no_asocia():
    """Chaleco horizontalmente lejos del centro de la cabeza no debe asociarse."""
    vest_off = Detection(class_id=CLASS_VEST, box=VEST_NOT_ALIGNED, score=0.9)
    report = verify_frame([_head_helmet(), vest_off])
    # El veredicto de la cabeza es NO_CONFORME por chaleco no visible.
    head_verdict = next(
        v for v in report.verdicts if v.head_box == HEAD_A
    )
    assert head_verdict.status == ComplianceStatus.NO_CONFORME
    assert "chaleco_no_visible" in head_verdict.reasons


def test_caso_07_chaleco_huerfano_es_indeterminado():
    """Chaleco sin cabeza asociada en la imagen -> INDETERMINADO."""
    report = verify_frame([_vest()])
    assert len(report.verdicts) == 1
    v = report.verdicts[0]
    assert v.status == ComplianceStatus.INDETERMINADO
    assert "cabeza_no_visible" in v.reasons


def test_caso_08_dos_trabajadores_conformes():
    """Dos cabezas con casco + dos chalecos correctamente apareados -> 2 CONFORMES."""
    dets = [
        _head_helmet(HEAD_A), _vest(VEST_A),
        _head_helmet(HEAD_B), _vest(VEST_B),
    ]
    report = verify_frame(dets)
    assert len(report.verdicts) == 2
    assert all(v.status == ComplianceStatus.CONFORME for v in report.verdicts)
    assert report.compliance_rate == pytest.approx(1.0)


def test_caso_09_tasa_de_cumplimiento_50_porciento():
    """Una cabeza con casco+chaleco y otra sin casco -> tasa 0.5."""
    dets = [
        _head_helmet(HEAD_A), _vest(VEST_A),                # CONFORME
        _head_no_helmet(HEAD_B), _vest(VEST_B),             # NO_CONFORME
    ]
    report = verify_frame(dets)
    assert len(report.verdicts) == 2
    assert report.compliance_rate == pytest.approx(0.5)


def test_caso_10_sin_detecciones_reporte_vacio():
    """Frame sin detecciones: reporte vacío y tasa neutra."""
    report = verify_frame([])
    assert report.verdicts == []
    assert report.compliance_rate == 0.0


def test_caso_11_no_vest_person_aislado_es_no_conforme():
    """Detección de `no_vest_person` sin cabeza ni `person` cercana -> NO_CONFORME."""
    report = verify_frame([_no_vest_person()])
    assert len(report.verdicts) == 1
    v = report.verdicts[0]
    assert v.status == ComplianceStatus.NO_CONFORME
    assert "chaleco_ausente" in v.reasons


def test_caso_12_no_vest_person_solapando_cabeza_no_duplica():
    """no_vest_person que solapa con una cabeza ya procesada no genera veredicto extra."""
    # no_vest_person que envuelve PERSON_A y por lo tanto solapa con HEAD_A
    nv_overlapping = Detection(class_id=CLASS_NO_VEST_PERSON, box=PERSON_A, score=0.9)
    report = verify_frame([_head_helmet(), _vest(), nv_overlapping])
    # Solo el veredicto de la cabeza (CONFORME), no se duplica con nv.
    assert len(report.verdicts) == 1
    assert report.verdicts[0].status == ComplianceStatus.CONFORME


def test_caso_13_person_bbox_se_usa_para_render_si_disponible():
    """Si hay `person` que contiene la cabeza, su bbox se usa para renderizar."""
    report = verify_frame([_head_helmet(), _vest(), _person()])
    v = report.verdicts[0]
    assert v.status == ComplianceStatus.CONFORME
    # person_box debería ser el bbox de la persona, no uno virtual.
    assert v.person_box == PERSON_A


def test_caso_14_chaleco_compite_y_gana_el_mas_cercano():
    """Si hay dos chalecos válidos, gana el más cercano debajo de la cabeza."""
    vest_close = Detection(class_id=CLASS_VEST, box=VEST_A, score=0.9)
    # Otro chaleco más abajo pero todavía dentro del umbral
    vest_below = Detection(
        class_id=CLASS_VEST, box=(110.0, 260.0, 190.0, 360.0), score=0.9
    )
    report = verify_frame([_head_helmet(), vest_close, vest_below])
    v_head = next(v for v in report.verdicts if v.head_box == HEAD_A)
    assert v_head.status == ComplianceStatus.CONFORME
    # Debe haber asociado el chaleco MÁS CERCANO (VEST_A).
    assert v_head.vest_box == VEST_A
    # El chaleco lejano queda huérfano -> INDETERMINADO.
    assert any(
        v.status == ComplianceStatus.INDETERMINADO for v in report.verdicts
    )


def test_caso_15_umbrales_configurables():
    """Subir el margen vertical permite asociar un chaleco antes descartado."""
    vest_far = Detection(class_id=CLASS_VEST, box=VEST_FAR_BELOW, score=0.9)

    cfg_estricto = VerifierConfig(max_vest_drop_in_head_heights=5.0)
    cfg_laxo = VerifierConfig(max_vest_drop_in_head_heights=20.0)

    r_estricto = verify_frame([_head_helmet(), vest_far], cfg_estricto)
    r_laxo = verify_frame([_head_helmet(), vest_far], cfg_laxo)

    head_estricto = next(v for v in r_estricto.verdicts if v.head_box == HEAD_A)
    head_laxo = next(v for v in r_laxo.verdicts if v.head_box == HEAD_A)

    assert head_estricto.status == ComplianceStatus.NO_CONFORME
    assert head_laxo.status == ComplianceStatus.CONFORME
