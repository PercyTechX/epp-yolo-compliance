"""Tests unitarios de las utilidades geométricas del verificador."""
from __future__ import annotations

import pytest

from src.compliance.geometry import (
    associate_head_to_person,
    associate_vest_to_person,
    box_area,
    containment,
    horizontal_overlap_ratio,
    in_upper_third,
    intersection_area,
    iou,
)


def test_iou_identical_boxes_es_uno():
    """Dos cajas idénticas deben tener IoU = 1."""
    box = (10.0, 20.0, 110.0, 220.0)
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_cajas_disjuntas_es_cero():
    """Cajas sin intersección deben dar IoU = 0."""
    a = (0.0, 0.0, 10.0, 10.0)
    b = (100.0, 100.0, 110.0, 110.0)
    assert iou(a, b) == 0.0


def test_iou_solapamiento_parcial_conocido():
    """IoU calculado manualmente sobre solapamiento parcial conocido."""
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 5.0, 15.0, 15.0)
    # intersección: 5x5 = 25; áreas: 100 y 100; unión: 175.
    assert iou(a, b) == pytest.approx(25.0 / 175.0)


def test_containment_inner_completamente_dentro():
    """Si `inner` está completo dentro de `outer`, la contención es 1."""
    inner = (10.0, 10.0, 20.0, 20.0)
    outer = (0.0, 0.0, 100.0, 100.0)
    assert containment(inner, outer) == pytest.approx(1.0)


def test_containment_caja_degenerada_es_cero():
    """Caja con área 0 debe dar contención 0 (sin divisiones por cero)."""
    inner = (10.0, 10.0, 10.0, 10.0)
    outer = (0.0, 0.0, 100.0, 100.0)
    assert containment(inner, outer) == 0.0


def test_in_upper_third_centro_arriba():
    """Una caja hija con centro en el tercio superior debe ser detectada."""
    parent = (0.0, 0.0, 100.0, 300.0)
    child = (40.0, 10.0, 60.0, 60.0)  # centro y = 35, tercio superior hasta y=100
    assert in_upper_third(child, parent)


def test_in_upper_third_centro_abajo():
    """Una caja hija con centro fuera del tercio superior debe rechazarse."""
    parent = (0.0, 0.0, 100.0, 300.0)
    child = (40.0, 200.0, 60.0, 280.0)
    assert not in_upper_third(child, parent)


def test_horizontal_overlap_ratio_proyeccion_completa():
    """Si `a` se proyecta enteramente dentro de `b`, el ratio es 1."""
    a = (10.0, 0.0, 20.0, 5.0)
    b = (0.0, 0.0, 100.0, 100.0)
    assert horizontal_overlap_ratio(a, b) == pytest.approx(1.0)


def test_horizontal_overlap_ratio_sin_solape():
    """Sin solapamiento en X, el ratio es 0."""
    a = (200.0, 0.0, 210.0, 5.0)
    b = (0.0, 0.0, 100.0, 100.0)
    assert horizontal_overlap_ratio(a, b) == 0.0


def test_associate_head_to_person_caso_positivo():
    """Cabeza en tercio superior y bien solapada en X debe asociarse."""
    person = (100.0, 50.0, 200.0, 350.0)   # 100x300
    head = (120.0, 60.0, 180.0, 100.0)     # arriba, centrada
    assert associate_head_to_person(head, person)


def test_associate_head_to_person_caso_negativo_por_posicion():
    """Cabeza muy abajo no debe asociarse aunque solape en X."""
    person = (100.0, 50.0, 200.0, 350.0)
    head = (120.0, 280.0, 180.0, 320.0)
    assert not associate_head_to_person(head, person)


def test_associate_vest_to_person_caso_positivo():
    """Chaleco contenido en la persona debe asociarse."""
    person = (100.0, 50.0, 200.0, 350.0)
    vest = (110.0, 150.0, 190.0, 250.0)
    assert associate_vest_to_person(vest, person)


def test_associate_vest_to_person_no_contenido():
    """Chaleco mayormente fuera de la persona debe rechazarse."""
    person = (100.0, 50.0, 200.0, 350.0)
    vest = (250.0, 150.0, 350.0, 250.0)
    assert not associate_vest_to_person(vest, person)


def test_box_area_y_intersection_area_basicas():
    """Sanity check de área y de intersección entre dos cajas conocidas."""
    a = (0.0, 0.0, 10.0, 20.0)
    b = (5.0, 5.0, 15.0, 25.0)
    assert box_area(a) == 200.0
    assert intersection_area(a, b) == 5.0 * 15.0  # 5 wide, 15 tall
