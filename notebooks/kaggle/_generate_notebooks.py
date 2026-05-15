"""Generador de notebooks scaffold para Kaggle.

Este script se ejecuta una sola vez para producir los .ipynb delgados
de la carpeta `notebooks/kaggle/`. Se conserva en el repo para que sea
trivial regenerarlos si cambian los hiperparámetros base.

Uso:
    python notebooks/kaggle/_generate_notebooks.py

Diseño de cada notebook (4 celdas):
    1. Markdown de título y rol del notebook.
    2. Setup: clonar repo, instalar dependencias.
    3. Carga de config y data.yaml apuntando al Kaggle Dataset.
    4. Entrenamiento + guardado de artefactos.

Toda la lógica vive en `src/` del repositorio. Los notebooks NO
contienen funciones; solo orquestan llamadas.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

OUT_DIR = Path(__file__).resolve().parent
REPO_URL_PLACEHOLDER = "https://github.com/<usuario>/<repo>.git"

NB_VERSION = 4


def md(*lines: str) -> dict:
    """Construye una celda markdown."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _to_source(lines),
    }


def code(*lines: str) -> dict:
    """Construye una celda de código."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _to_source(lines),
    }


def _to_source(lines: List[str]) -> List[str]:
    """Une líneas con saltos respetando el formato nbformat."""
    joined = "\n".join(lines)
    return [line + "\n" for line in joined.split("\n")][:-1] + [joined.split("\n")[-1]]


def base_notebook(cells: List[dict]) -> dict:
    """Construye el dict base de un notebook nbformat 4."""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": NB_VERSION,
        "nbformat_minor": 5,
    }


# ---------------------------------------------------------------------------
# Celdas comunes.
# ---------------------------------------------------------------------------


def cell_setup() -> dict:
    """Celda de setup: clonar repo e instalar dependencias en Kaggle."""
    return code(
        "# Setup en Kaggle: clona el repo del proyecto e instala dependencias.",
        "# El código pesado vive en `src/`; este notebook solo orquesta.",
        "import os, subprocess, sys",
        "",
        f'REPO_URL = "{REPO_URL_PLACEHOLDER}"',
        'REPO_DIR = "/kaggle/working/repo"',
        "",
        "if not os.path.isdir(REPO_DIR):",
        '    subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR])',
        "os.chdir(REPO_DIR)",
        'subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])',
    )


def cell_load_config(hyp_file: str) -> dict:
    """Celda que carga el hyp .yaml y apunta data.yaml al Kaggle Dataset."""
    return code(
        f'# Carga el archivo de hiperparámetros del experimento ({hyp_file}).',
        "# Sobrescribe `path` de data.yaml para apuntar al Kaggle Dataset montado.",
        "import yaml, shutil",
        "from pathlib import Path",
        "",
        f'HYP_FILE = Path("configs/{hyp_file}")',
        'DATA_YAML = Path("configs/data.yaml")',
        '# TODO: reemplazar por la ruta real del Kaggle Dataset montado, por ejemplo:',
        '#   /kaggle/input/epp-yolo-processed',
        'KAGGLE_DATA_ROOT = "/kaggle/input/epp-yolo-processed"',
        "",
        "with HYP_FILE.open() as f:",
        "    hyp = yaml.safe_load(f)",
        "",
        "data_cfg = yaml.safe_load(DATA_YAML.read_text())",
        'data_cfg["path"] = KAGGLE_DATA_ROOT',
        'patched_data = Path("/kaggle/working/data_patched.yaml")',
        'patched_data.write_text(yaml.safe_dump(data_cfg, sort_keys=False))',
        "",
        'print("Hiperparámetros:", hyp)',
        'print("data.yaml parcheado:", patched_data)',
    )


def cell_train(experiment_name: str) -> dict:
    """Celda de entrenamiento con Ultralytics."""
    return code(
        "# Entrenamiento. Usa la API de Ultralytics directamente; toda la",
        "# justificación de hiperparámetros vive en el archivo YAML cargado.",
        "from ultralytics import YOLO",
        "",
        'model = YOLO(hyp["model"])',
        "results = model.train(",
        "    data=str(patched_data),",
        f'    name="{experiment_name}",',
        '    project="/kaggle/working/runs",',
        "    **{k: v for k, v in hyp.items() if k != 'model'},",
        ")",
        "",
        "# Los artefactos quedan en /kaggle/working/runs/<name>/. Descargar el",
        "# best.pt al final del run y publicarlo como release del repo Git.",
    )


def cell_save_artifacts(experiment_name: str) -> dict:
    """Celda que enumera artefactos para descarga manual al terminar."""
    return code(
        "# Listar artefactos producidos para descarga (no commitear los .pt).",
        "from pathlib import Path",
        f'run_dir = Path("/kaggle/working/runs/{experiment_name}")',
        "for p in sorted(run_dir.rglob('*')):",
        '    if p.is_file():',
        '        print(p)',
    )


# ---------------------------------------------------------------------------
# Definición de los 6 notebooks.
# ---------------------------------------------------------------------------


NOTEBOOKS = [
    {
        "filename": "01_baseline_train.ipynb",
        "title": "Baseline (Fase 2): YOLOv8s @ 640",
        "description": (
            "Entrenamiento baseline con configuración por defecto. Establece "
            "la referencia honesta para comparar los experimentos de la fase 3. "
            "Hiperparámetros en `configs/hyp_baseline.yaml`."
        ),
        "hyp": "hyp_baseline.yaml",
        "exp_name": "baseline",
    },
    {
        "filename": "02_exp_A_resolution.ipynb",
        "title": "Experimento A (Fase 3): resolución 960",
        "description": (
            "Sube la resolución de entrada de 640 a 960 manteniendo el resto "
            "del baseline. Objetivo: mejorar detección de objetos pequeños. "
            "Hiperparámetros en `configs/hyp_exp_A.yaml`."
        ),
        "hyp": "hyp_exp_A.yaml",
        "exp_name": "exp_A_resolution",
    },
    {
        "filename": "03_exp_B_augmentation.ipynb",
        "title": "Experimento B (Fase 3): Random Erasing + MixUp",
        "description": (
            "Añade Random Erasing y MixUp moderado para simular oclusión y "
            "ganar robustez. Hiperparámetros en `configs/hyp_exp_B.yaml`."
        ),
        "hyp": "hyp_exp_B.yaml",
        "exp_name": "exp_B_augmentation",
    },
    {
        "filename": "04_exp_C_class_weights.ipynb",
        "title": "Experimento C (Fase 3): pesos de clase / oversampling",
        "description": (
            "Ataca el desbalance de clases minoritarias "
            "(`head_without_helmet`, `no_vest_person`). "
            "La estrategia concreta (oversampling de manifiesto o ajuste "
            "manual de la cls loss) se decide en este notebook. "
            "Hiperparámetros base en `configs/hyp_exp_C.yaml`."
        ),
        "hyp": "hyp_exp_C.yaml",
        "exp_name": "exp_C_class_weights",
    },
    {
        "filename": "05_exp_D_close_mosaic.ipynb",
        "title": "Experimento D (Fase 3): close_mosaic en los últimos 15 epochs",
        "description": (
            "Desactiva el mosaico en los últimos 15 epochs para afinar el "
            "modelo con imágenes en su distribución natural. "
            "Hiperparámetros en `configs/hyp_exp_D.yaml`."
        ),
        "hyp": "hyp_exp_D.yaml",
        "exp_name": "exp_D_close_mosaic",
    },
    {
        "filename": "06_inference_demo.ipynb",
        "title": "Demo de inferencia (Fase 6): pipeline YOLO + verificador",
        "description": (
            "Ejecuta el modelo final sobre un video corto no visto en el "
            "entrenamiento y aplica el verificador determinista. Exporta un "
            "video anotado con bboxes coloreados por estado y tasa de "
            "cumplimiento por fotograma."
        ),
        "hyp": None,
        "exp_name": "demo",
    },
]


def build_train_notebook(spec: dict) -> dict:
    """Construye un notebook de entrenamiento (notebooks 01-05)."""
    cells = [
        md(
            f"# {spec['title']}",
            "",
            spec["description"],
            "",
            "> **Notebook delgado**: solo orquesta. La lógica vive en `src/` del repositorio.",
        ),
        cell_setup(),
        cell_load_config(spec["hyp"]),
        cell_train(spec["exp_name"]),
        cell_save_artifacts(spec["exp_name"]),
    ]
    return base_notebook(cells)


def build_demo_notebook(spec: dict) -> dict:
    """Construye el notebook de demo de inferencia (notebook 06)."""
    inference_cell = code(
        "# Inferencia frame a frame sobre un video corto + verificador determinista.",
        "# El video DEBE ser no visto durante el entrenamiento.",
        "import cv2",
        "from pathlib import Path",
        "from ultralytics import YOLO",
        "from src.compliance.verifier import Detection, verify_frame",
        "from src.compliance.renderer import annotate_frame",
        "",
        '# TODO: descargar el best.pt del release de Git y subirlo como Kaggle Dataset,',
        '# o subir el archivo manualmente al notebook antes de ejecutar esta celda.',
        'WEIGHTS = "/kaggle/input/epp-yolo-weights/best.pt"',
        'INPUT_VIDEO = "/kaggle/input/epp-demo-videos/demo.mp4"',
        'OUTPUT_VIDEO = "/kaggle/working/demo_annotated.mp4"',
        "",
        "model = YOLO(WEIGHTS)",
        "",
        "cap = cv2.VideoCapture(INPUT_VIDEO)",
        "fps = cap.get(cv2.CAP_PROP_FPS) or 25.0",
        "w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))",
        "h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))",
        'fourcc = cv2.VideoWriter_fourcc(*"mp4v")',
        "writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))",
        "",
        "while True:",
        "    ok, frame = cap.read()",
        "    if not ok:",
        "        break",
        "    results = model.predict(frame, verbose=False)[0]",
        "    dets = []",
        "    for box, cls, conf in zip(results.boxes.xyxy.cpu().numpy(),",
        "                              results.boxes.cls.cpu().numpy(),",
        "                              results.boxes.conf.cpu().numpy()):",
        "        dets.append(Detection(class_id=int(cls), box=tuple(box.tolist()), score=float(conf)))",
        "    report = verify_frame(dets)",
        "    annotated = annotate_frame(frame, report)",
        "    writer.write(annotated)",
        "",
        "cap.release()",
        "writer.release()",
        'print(f"Video anotado escrito en {OUTPUT_VIDEO}")',
    )
    cells = [
        md(
            f"# {spec['title']}",
            "",
            spec["description"],
            "",
            "> **Notebook delgado**: solo orquesta. La lógica vive en `src/` del repositorio.",
        ),
        cell_setup(),
        inference_cell,
    ]
    return base_notebook(cells)


def main() -> None:
    """Escribe los 6 notebooks scaffold en disco."""
    for spec in NOTEBOOKS:
        if spec["hyp"] is None:
            nb = build_demo_notebook(spec)
        else:
            nb = build_train_notebook(spec)
        out_path = OUT_DIR / spec["filename"]
        out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[notebooks] escrito: {out_path}")


if __name__ == "__main__":
    main()
