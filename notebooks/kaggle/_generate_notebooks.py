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
REPO_URL_PLACEHOLDER = "https://github.com/PercyTechX/epp-yolo-compliance.git"

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
        "# La ruta se auto-detecta porque Kaggle a veces monta los datasets",
        "# bajo /kaggle/input/<slug>/ y otras bajo /kaggle/input/datasets/<user>/<slug>/.",
        "import os, yaml",
        "from pathlib import Path",
        "",
        "",
        "def find_dataset_root() -> str:",
        '    """Localiza la raíz del Kaggle Dataset buscando images/ + labels/."""',
        "    candidates = [",
        '        "/kaggle/input/epp-yolo-processed",',
        '        "/kaggle/input/datasets/jpnazariop/epp-yolo-processed",',
        "    ]",
        "    for c in candidates:",
        '        if os.path.isdir(f"{c}/images/train"):',
        "            return c",
        "    # Fallback: escaneo recursivo de /kaggle/input/.",
        '    for root, dirs, _ in os.walk("/kaggle/input"):',
        '        if "images" in dirs and "labels" in dirs:',
        '            if os.path.isdir(f"{root}/images/train"):',
        "                return root",
        '    raise RuntimeError("Dataset no encontrado bajo /kaggle/input/")',
        "",
        "",
        f'HYP_FILE = Path("configs/{hyp_file}")',
        'DATA_YAML = Path("configs/data.yaml")',
        "KAGGLE_DATA_ROOT = find_dataset_root()",
        'print(f"Dataset montado en: {KAGGLE_DATA_ROOT}")',
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


def cell_finetune() -> dict:
    """Celda principal de fine-tuning del modelo D sobre OtroEnfoque."""
    return code(
        "# Fine-tuning del modelo D sobre el dataset OtroEnfoque remapeado.",
        "# Continúa el entrenamiento desde best.pt existente con lr más bajo",
        "# para incorporar la riqueza de la clase `person` sin destruir lo",
        "# aprendido sobre casco/chaleco.",
        "import os, yaml",
        "from pathlib import Path",
        "from ultralytics import YOLO",
        "",
        "",
        "def find_otroenfoque_root() -> str:",
        '    """Localiza la raíz del dataset OtroEnfoque (debe tener images/train + labels/train)."""',
        "    # Buscamos específicamente el dataset OtroEnfoque por el path",
        "    # para no confundirlo con el dataset principal epp-yolo-processed.",
        '    for root, dirs, _ in os.walk("/kaggle/input"):',
        '        path_str = root.lower()',
        '        if ("otroenfoque" in path_str or "remapped" in path_str) and \\',
        '           "images" in dirs and "labels" in dirs and \\',
        '           os.path.isdir(f"{root}/images/train"):',
        "            return root",
        '    raise RuntimeError("Dataset OtroEnfoque remapped no encontrado bajo /kaggle/input/")',
        "",
        "",
        "def find_best_pt() -> str:",
        '    """Localiza el best.pt del modelo D existente bajo /kaggle/input/."""',
        '    for root, _, files in os.walk("/kaggle/input"):',
        '        if "best.pt" in files:',
        '            return os.path.join(root, "best.pt")',
        '    raise RuntimeError("best.pt no encontrado bajo /kaggle/input/")',
        "",
        "",
        "DATA_ROOT = find_otroenfoque_root()",
        "WEIGHTS = find_best_pt()",
        'print(f"Dataset OtroEnfoque: {DATA_ROOT}")',
        'print(f"Pesos iniciales:    {WEIGHTS}")',
        "",
        "# Construir data.yaml apuntando al dataset OtroEnfoque montado.",
        "# No tiene `test`, solo train/val.",
        "data_cfg = {",
        '    "path": DATA_ROOT,',
        '    "train": "images/train",',
        '    "val": "images/val",',
        '    "names": {',
        '        0: "head_with_helmet",',
        '        1: "head_without_helmet",',
        '        2: "person",',
        '        3: "vest",',
        '        4: "no_vest_person",',
        "    },",
        "}",
        'patched = Path("/kaggle/working/data_finetune.yaml")',
        'patched.write_text(yaml.safe_dump(data_cfg, sort_keys=False))',
        'print(f"data.yaml para fine-tune: {patched}")',
        "",
        "# Cargar el modelo desde best.pt y continuar entrenando.",
        "# Ultralytics auto-detecta que ya hay pesos y hace transfer learning.",
        "model = YOLO(WEIGHTS)",
        "",
        "results = model.train(",
        "    data=str(patched),",
        '    name="finetune_otroenfoque",',
        '    project="/kaggle/working/runs",',
        "    # Configuración de fine-tuning conservadora:",
        "    epochs=25,",
        "    patience=10,",
        "    imgsz=640,",
        "    batch=16,",
        '    optimizer="SGD",',
        "    lr0=0.001,           # 10x menor que el lr original (0.01).",
        "    lrf=0.01,",
        "    momentum=0.937,",
        "    weight_decay=0.0005,",
        "    warmup_epochs=2.0,",
        "    # Augmentaciones suaves (mantener lo aprendido):",
        "    mosaic=1.0,",
        "    mixup=0.0,",
        "    erasing=0.0,",
        "    close_mosaic=8,      # apaga mosaic en los últimos 8 de 25 epochs.",
        "    fliplr=0.5,",
        "    hsv_h=0.015,",
        "    hsv_s=0.7,",
        "    hsv_v=0.4,",
        "    translate=0.1,",
        "    scale=0.5,",
        ")",
        "",
        'best_pt_path = "/kaggle/working/runs/finetune_otroenfoque/weights/best.pt"',
        'print(f"\\nModelo fine-tuned guardado en: {best_pt_path}")',
        "",
        "# Validación final del modelo fine-tuned sobre el val de OtroEnfoque.",
        'print("\\n=== Validación final sobre val de OtroEnfoque ===")',
        "metrics = model.val(data=str(patched), verbose=False)",
        'print(f"mAP@0.5      global: {metrics.box.map50:.4f}")',
        'print(f"mAP@0.5-0.95 global: {metrics.box.map:.4f}")',
        'print(f"Precision    global: {metrics.box.mp:.4f}")',
        'print(f"Recall       global: {metrics.box.mr:.4f}")',
    )


def cell_occlusion_analysis() -> dict:
    """Celda de análisis de robustez por nivel de oclusión sobre test."""
    return code(
        "# Análisis de oclusión: particiona el test set por nivel de oclusión y",
        "# corre validación independiente sobre cada subset.",
        "import os, json, yaml",
        "from pathlib import Path",
        "from ultralytics import YOLO",
        "from src.analysis.occlusion import partition_by_occlusion",
        "",
        "",
        "def find_dataset_root() -> str:",
        '    """Localiza la raíz del Kaggle Dataset processed."""',
        '    for root, dirs, _ in os.walk("/kaggle/input"):',
        '        if "images" in dirs and "labels" in dirs:',
        '            if os.path.isdir(f"{root}/images/test"):',
        "                return root",
        '    raise RuntimeError("Dataset processed no encontrado bajo /kaggle/input/")',
        "",
        "",
        "def find_weights() -> str:",
        '    """Localiza un best.pt bajo /kaggle/input/."""',
        '    for root, _, files in os.walk("/kaggle/input"):',
        '        if "best.pt" in files:',
        '            return os.path.join(root, "best.pt")',
        '    raise RuntimeError("best.pt no encontrado")',
        "",
        "",
        "DATA_ROOT = Path(find_dataset_root())",
        "WEIGHTS = find_weights()",
        'LABELS_TEST = DATA_ROOT / "labels" / "test"',
        'IMAGES_TEST = DATA_ROOT / "images" / "test"',
        'print(f"Dataset: {DATA_ROOT}")',
        'print(f"Pesos:   {WEIGHTS}")',
        "",
        "# Particionar test set por oclusión.",
        "partitions = partition_by_occlusion(LABELS_TEST, low_threshold=0.15, high_threshold=0.40)",
        'print(f"\\nDistribución por nivel de oclusión:")',
        "for level, stems in partitions.items():",
        '    print(f"  {level:7s}: {len(stems):4d} imágenes")',
        "",
        "# Escribir un .txt por nivel con paths absolutos a las imágenes.",
        "list_files = {}",
        'for level, stems in partitions.items():',
        "    if not stems:",
        "        continue",
        "    paths = []",
        "    for stem in stems:",
        '        for ext in [".jpg", ".jpeg", ".png"]:',
        '            cand = IMAGES_TEST / f"{stem}{ext}"',
        "            if cand.exists():",
        "                paths.append(str(cand))",
        "                break",
        '    list_path = Path(f"/kaggle/working/occlusion_{level}.txt")',
        '    list_path.write_text("\\n".join(paths))',
        "    list_files[level] = list_path",
        "",
        "# Cargar modelo una sola vez.",
        "model = YOLO(WEIGHTS)",
        "",
        "# Correr val() por cada nivel sobrescribiendo `val` con el txt list.",
        'data_cfg_base = yaml.safe_load(Path("configs/data.yaml").read_text())',
        'data_cfg_base["path"] = str(DATA_ROOT)',
        "",
        "results = {}",
        "for level, list_path in list_files.items():",
        '    print(f"\\n=== Validando nivel {level} ({len(partitions[level])} imágenes) ===")',
        "    data_cfg = dict(data_cfg_base)",
        '    data_cfg["val"] = str(list_path)',
        '    patched = Path(f"/kaggle/working/data_occlusion_{level}.yaml")',
        "    patched.write_text(yaml.safe_dump(data_cfg, sort_keys=False))",
        "    metrics = model.val(",
        "        data=str(patched),",
        '        split="val",',
        "        project=\"/kaggle/working/runs\",",
        "        name=f\"occlusion_{level}\",",
        "        save_json=False,",
        "        plots=False,",
        "        verbose=False,",
        "    )",
        "    results[level] = {",
        '        "n_images": len(partitions[level]),',
        '        "mAP50": float(metrics.box.map50),',
        '        "mAP50_95": float(metrics.box.map),',
        '        "precision": float(metrics.box.mp),',
        '        "recall": float(metrics.box.mr),',
        "    }",
        "",
        'Path("/kaggle/working/occlusion_results.json").write_text(json.dumps(results, indent=2))',
        'print("\\n=== Resumen final ===")',
        'print(json.dumps(results, indent=2))',
    )


def cell_eval_test_set() -> dict:
    """Celda de evaluación del modelo entrenado sobre el split de test."""
    return code(
        "# Evaluación final del modelo elegido sobre el split test (no visto).",
        "# Auto-detecta tanto la ruta del Kaggle Dataset como el best.pt en /kaggle/input/.",
        "import os, yaml, json",
        "from pathlib import Path",
        "from ultralytics import YOLO",
        "",
        "",
        "def find_dataset_root() -> str:",
        '    """Localiza la raíz del Kaggle Dataset processed (images/+labels/)."""',
        '    for root, dirs, _ in os.walk("/kaggle/input"):',
        '        if "images" in dirs and "labels" in dirs:',
        '            if os.path.isdir(f"{root}/images/test"):',
        "                return root",
        '    raise RuntimeError("Dataset processed no encontrado bajo /kaggle/input/")',
        "",
        "",
        "def find_weights() -> str:",
        '    """Localiza un best.pt cualquiera bajo /kaggle/input/."""',
        '    for root, _, files in os.walk("/kaggle/input"):',
        '        if "best.pt" in files:',
        '            return os.path.join(root, "best.pt")',
        '    raise RuntimeError("best.pt no encontrado bajo /kaggle/input/")',
        "",
        "",
        "DATA_ROOT = find_dataset_root()",
        "WEIGHTS = find_weights()",
        'print(f"Dataset montado en: {DATA_ROOT}")',
        'print(f"Pesos:               {WEIGHTS}")',
        "",
        "# Parchear data.yaml con la ruta real de Kaggle.",
        'data_cfg = yaml.safe_load(Path("configs/data.yaml").read_text())',
        'data_cfg["path"] = DATA_ROOT',
        'patched = Path("/kaggle/working/data_patched.yaml")',
        'patched.write_text(yaml.safe_dump(data_cfg, sort_keys=False))',
        "",
        "model = YOLO(WEIGHTS)",
        "metrics = model.val(",
        "    data=str(patched),",
        '    split="test",',
        '    project="/kaggle/working/runs",',
        '    name="test_eval_exp_D",',
        "    save_json=True,",
        "    plots=True,",
        ")",
        "",
        "# Resumen estructurado para descargar.",
        "summary = {",
        '    "mAP50": float(metrics.box.map50),',
        '    "mAP50_95": float(metrics.box.map),',
        '    "precision": float(metrics.box.mp),',
        '    "recall": float(metrics.box.mr),',
        '    "per_class": {',
        '        str(cid): {',
        '            "mAP50": float(metrics.box.maps[cid] if cid < len(metrics.box.maps) else 0.0),',
        '        } for cid in range(len(data_cfg["names"]))',
        "    },",
        "}",
        'Path("/kaggle/working/test_metrics.json").write_text(json.dumps(summary, indent=2))',
        'print(json.dumps(summary, indent=2))',
    )


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
    {
        "filename": "07_test_evaluation.ipynb",
        "title": "Evaluación final (Fase 5): modelo D sobre test set",
        "description": (
            "Carga el modelo final entrenado (best.pt subido como Kaggle Dataset) "
            "y lo evalúa sobre el split test del dataset epp-yolo-processed. "
            "Reporta mAP global y por clase + matriz de confusión + curvas P/R."
        ),
        "hyp": None,
        "exp_name": "test_eval",
        "is_eval": True,
    },
    {
        "filename": "08_occlusion_analysis.ipynb",
        "title": "Análisis de robustez por nivel de oclusión (Fase 5.2)",
        "description": (
            "Particiona el test set en 3 niveles de oclusión (low / medium / high) "
            "según el IoU máximo entre cajas en cada imagen, y reporta mAP global "
            "y por clase para cada nivel. Sirve para el entregable 3 del enunciado."
        ),
        "hyp": None,
        "exp_name": "occlusion",
        "is_occlusion": True,
    },
    {
        "filename": "09_finetune_otroenfoque.ipynb",
        "title": "Fine-tuning del modelo D sobre OtroEnfoque (Mejora iterativa)",
        "description": (
            "Continúa el entrenamiento del modelo D sobre el dataset "
            "Construction Site Safety remapeado (OtroEnfoque). Objetivo: "
            "mejorar la clase `person` (mAP 0.17 -> esperado 0.6+) sin "
            "perder lo aprendido sobre casco y chaleco. Usa learning rate "
            "10x menor (0.001) y 25 epochs."
        ),
        "hyp": None,
        "exp_name": "finetune",
        "is_finetune": True,
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


def build_eval_notebook(spec: dict) -> dict:
    """Construye el notebook de evaluación sobre test set (notebook 07)."""
    cells = [
        md(
            f"# {spec['title']}",
            "",
            spec["description"],
            "",
            "> **Notebook delgado**: solo orquesta. La lógica vive en `src/` del repositorio.",
            "",
            "**Requisitos:**",
            "- Adjuntar el dataset `epp-yolo-processed` como Input.",
            "- Adjuntar un dataset que contenga el `best.pt` del modelo a evaluar.",
        ),
        cell_setup(),
        cell_eval_test_set(),
    ]
    return base_notebook(cells)


def build_occlusion_notebook(spec: dict) -> dict:
    """Construye el notebook de análisis de oclusión (notebook 08)."""
    cells = [
        md(
            f"# {spec['title']}",
            "",
            spec["description"],
            "",
            "> **Notebook delgado**: solo orquesta. La lógica vive en `src/` del repositorio.",
            "",
            "**Requisitos:**",
            "- Adjuntar el dataset `epp-yolo-processed` como Input.",
            "- Adjuntar un dataset que contenga el `best.pt` del modelo a evaluar.",
        ),
        cell_setup(),
        cell_occlusion_analysis(),
    ]
    return base_notebook(cells)


def build_finetune_notebook(spec: dict) -> dict:
    """Construye el notebook de fine-tuning sobre OtroEnfoque (notebook 09)."""
    cells = [
        md(
            f"# {spec['title']}",
            "",
            spec["description"],
            "",
            "> **Notebook delgado**: solo orquesta. La lógica vive en `src/` del repositorio.",
            "",
            "**Requisitos:**",
            "- Adjuntar el dataset `epp-yolo-otroenfoque-remapped` como Input.",
            "- Adjuntar el dataset `epp-yolo-exp-d-weights` (con el `best.pt` del modelo D).",
            "- GPU: T4 x2 o P100.",
            "- Internet: On.",
            "",
            "**Tiempo estimado:** 2-3 horas en T4 x2 / 1.5-2 h en P100.",
        ),
        cell_setup(),
        cell_finetune(),
    ]
    return base_notebook(cells)


def main() -> None:
    """Escribe los notebooks scaffold en disco."""
    for spec in NOTEBOOKS:
        if spec.get("is_finetune"):
            nb = build_finetune_notebook(spec)
        elif spec.get("is_occlusion"):
            nb = build_occlusion_notebook(spec)
        elif spec.get("is_eval"):
            nb = build_eval_notebook(spec)
        elif spec["hyp"] is None:
            nb = build_demo_notebook(spec)
        else:
            nb = build_train_notebook(spec)
        out_path = OUT_DIR / spec["filename"]
        out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[notebooks] escrito: {out_path}")


if __name__ == "__main__":
    main()
