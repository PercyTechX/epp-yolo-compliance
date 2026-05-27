"""Tests del módulo de análisis de oclusión."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.analysis.occlusion import (
    compute_max_iou,
    occlusion_level,
    partition_by_occlusion,
    read_yolo_bboxes,
)


def test_occlusion_level_baja():
    """Solapamientos pequeños -> 'low'."""
    assert occlusion_level(0.0) == "low"
    assert occlusion_level(0.10) == "low"
    assert occlusion_level(0.149) == "low"


def test_occlusion_level_media():
    """Solapamientos medios -> 'medium'."""
    assert occlusion_level(0.15) == "medium"
    assert occlusion_level(0.30) == "medium"
    assert occlusion_level(0.399) == "medium"


def test_occlusion_level_alta():
    """Solapamientos grandes -> 'high'."""
    assert occlusion_level(0.40) == "high"
    assert occlusion_level(0.85) == "high"


def test_compute_max_iou_sin_pares():
    """0 o 1 caja debe dar IoU máximo 0."""
    assert compute_max_iou([]) == 0.0
    assert compute_max_iou([(0.0, 0.0, 0.5, 0.5)]) == 0.0


def test_compute_max_iou_pares_disjuntos():
    """Cajas sin intersección -> IoU 0."""
    a = (0.0, 0.0, 0.2, 0.2)
    b = (0.5, 0.5, 0.7, 0.7)
    assert compute_max_iou([a, b]) == 0.0


def test_compute_max_iou_un_par_solapado():
    """IoU conocido entre dos cajas idénticas -> 1."""
    a = (0.2, 0.2, 0.4, 0.4)
    assert compute_max_iou([a, a]) == pytest.approx(1.0)


def test_compute_max_iou_devuelve_el_maximo():
    """Cuando hay varios pares, devuelve el de mayor IoU."""
    a = (0.0, 0.0, 0.2, 0.2)
    b = (0.5, 0.5, 0.7, 0.7)
    c = (0.55, 0.55, 0.75, 0.75)  # solapa fuerte con b
    # b ∩ c es un rectángulo 0.15x0.15 = 0.0225, áreas 0.04 c/u, unión = 0.0575.
    # IoU(b,c) ≈ 0.391; IoU(a,b) = 0; IoU(a,c) = 0.
    assert compute_max_iou([a, b, c]) == pytest.approx(0.0225 / (0.04 + 0.04 - 0.0225))


def test_read_yolo_bboxes_lee_y_parsea(tmp_path: Path):
    """Lee un archivo .txt YOLO y retorna xyxy normalizado."""
    f = tmp_path / "x.txt"
    f.write_text("0 0.5 0.5 0.4 0.4\n2 0.2 0.2 0.1 0.1\n")
    bboxes = read_yolo_bboxes(f)
    assert len(bboxes) == 2
    assert bboxes[0] == pytest.approx((0.3, 0.3, 0.7, 0.7))
    assert bboxes[1] == pytest.approx((0.15, 0.15, 0.25, 0.25))


def test_read_yolo_bboxes_ignora_lineas_malas(tmp_path: Path):
    """Lineas vacías y mal formadas no rompen la lectura."""
    f = tmp_path / "x.txt"
    f.write_text("0 0.5 0.5 0.4 0.4\n\nclase_mal 0.0 0.0 0.1\n1 0.1 0.1 0.05 0.05\n")
    bboxes = read_yolo_bboxes(f)
    assert len(bboxes) == 2


def test_partition_by_occlusion_clasifica(tmp_path: Path):
    """Particiona un directorio de etiquetas según oclusión por imagen."""
    labels = tmp_path / "labels"
    labels.mkdir()
    # Imagen 1: dos cajas separadas -> low
    (labels / "img_low.txt").write_text("0 0.1 0.1 0.1 0.1\n1 0.9 0.9 0.1 0.1\n")
    # Imagen 2: dos cajas con IoU medio (~0.20)
    (labels / "img_medium.txt").write_text(
        "0 0.5 0.5 0.4 0.4\n1 0.6 0.6 0.4 0.4\n"
    )
    # Imagen 3: dos cajas iguales -> alta (IoU 1.0)
    (labels / "img_high.txt").write_text(
        "0 0.5 0.5 0.4 0.4\n1 0.5 0.5 0.4 0.4\n"
    )
    # Imagen 4: vacía -> ignorada
    (labels / "img_empty.txt").write_text("")

    partitions = partition_by_occlusion(labels)

    assert "img_low" in partitions["low"]
    assert "img_medium" in partitions["medium"]
    assert "img_high" in partitions["high"]
    # Total de stems clasificados (sin contar el vacío)
    total = sum(len(v) for v in partitions.values())
    assert total == 3
