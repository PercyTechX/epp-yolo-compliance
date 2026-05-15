"""Remapea las etiquetas de un dataset original a la taxonomía unificada.

Lee `configs/class_mapping.yaml` para obtener la tabla de mapeo del dataset
indicado y produce una copia de las etiquetas con los ids unificados,
preservando las imágenes (se copian o enlazan, no se modifican).

Uso típico:
    python -m src.data.remap_labels --dataset shwd \
        --src data/raw/shwd --dst data/interim/shwd

Layouts de entrada soportados:

  1) Plano (un solo directorio):
        <src>/images/*.{jpg,jpeg,png}
        <src>/labels/*.txt

  2) Roboflow (split ya separado):
        <src>/{train,valid,test}/images/*.{jpg,jpeg,png}
        <src>/{train,valid,test}/labels/*.txt

En ambos casos la salida es plana en `<dst>/images/` y `<dst>/labels/`.
En layout Roboflow los nombres se prefijan con el nombre del split de
origen (`train__foo.jpg`, `valid__foo.jpg`) para evitar colisiones.

Si el dataset original viene con los ids ya como enteros, este script
necesita además un mapa `id -> nombre` por dataset. Para mantener una sola
tabla de verdad, ese mapa se infiere del `data.yaml` del dataset original
(opción `--names-yaml`); si no se provee, se intenta leer `<src>/data.yaml`.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# Ruta al repo (asumiendo que este archivo vive en src/data/).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING = REPO_ROOT / "configs" / "class_mapping.yaml"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Remapea etiquetas YOLO de un dataset original a la taxonomía "
            "unificada del proyecto (5 clases)."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["shwd", "helmet", "ppe"],
        help="Nombre lógico del dataset (debe existir en class_mapping.yaml).",
    )
    parser.add_argument(
        "--src",
        required=True,
        type=Path,
        help="Carpeta raíz del dataset original (contiene images/ y labels/).",
    )
    parser.add_argument(
        "--dst",
        required=True,
        type=Path,
        help="Carpeta destino para el dataset remapeado.",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_MAPPING,
        help="Ruta al class_mapping.yaml (por defecto: configs/class_mapping.yaml).",
    )
    parser.add_argument(
        "--names-yaml",
        type=Path,
        default=None,
        help=(
            "data.yaml del dataset original (para resolver id -> nombre). "
            "Si se omite, se busca <src>/data.yaml."
        ),
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copia las imágenes al destino (por defecto se enlazan/relativizan).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No escribe nada; solo reporta lo que haría.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    """Lee un YAML y retorna su contenido como dict."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_id_to_name(
    src: Path, names_yaml: Optional[Path]
) -> Optional[Dict[int, str]]:
    """Resuelve el mapa id -> nombre del dataset original.

    Devuelve None si no se encuentra; en ese caso el script trata los ids
    del dataset original como ya alineados con los nombres de class_mapping
    (poco frecuente, pero útil para datasets sin metadatos).
    """
    candidate = names_yaml if names_yaml else src / "data.yaml"
    if not candidate.exists():
        return None
    cfg = load_yaml(candidate)
    names = cfg.get("names")
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    if isinstance(names, dict):
        # Ultralytics admite tanto lista como dict de ids -> nombres.
        return {int(k): v for k, v in names.items()}
    return None


def build_id_translation(
    dataset_name: str,
    mapping_cfg: dict,
    id_to_name: Optional[Dict[int, str]],
) -> Tuple[Dict[int, int], List[str]]:
    """Construye el mapa `id_original -> id_unificado`.

    Retorna también la lista de nombres originales descartados (para log).
    """
    if "datasets" not in mapping_cfg or dataset_name not in mapping_cfg["datasets"]:
        raise KeyError(
            f"El dataset '{dataset_name}' no está declarado en el class_mapping.yaml."
        )
    unified = mapping_cfg["unified_classes"]
    # Invertir unified: nombre -> id_unificado.
    name_to_unified_id = {v: int(k) for k, v in unified.items()}

    dataset_block = mapping_cfg["datasets"][dataset_name]
    name_mapping: Dict[str, str] = dataset_block.get("mapping", {}) or {}
    discard: List[str] = dataset_block.get("discard_classes", []) or []

    translation: Dict[int, int] = {}
    discarded_names: List[str] = []

    if id_to_name is None:
        # No tenemos ids del dataset original: el script no puede traducir.
        raise RuntimeError(
            "No se pudo resolver id -> nombre del dataset original. "
            "Provee --names-yaml apuntando al data.yaml del dataset."
        )

    for orig_id, orig_name in id_to_name.items():
        if orig_name in discard:
            discarded_names.append(orig_name)
            continue
        if orig_name not in name_mapping:
            # No está mapeado: lo descartamos por seguridad y advertimos.
            discarded_names.append(orig_name)
            continue
        unified_name = name_mapping[orig_name]
        if unified_name not in name_to_unified_id:
            raise ValueError(
                f"El nombre unificado '{unified_name}' no existe en "
                "unified_classes."
            )
        translation[int(orig_id)] = name_to_unified_id[unified_name]

    return translation, discarded_names


def remap_label_file(
    src_file: Path, dst_file: Path, translation: Dict[int, int]
) -> Tuple[int, int]:
    """Reescribe un archivo de etiquetas YOLO con los ids unificados.

    Retorna (líneas_escritas, líneas_descartadas).
    """
    written = 0
    dropped = 0
    out_lines: List[str] = []
    with src_file.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            try:
                orig_id = int(parts[0])
            except (ValueError, IndexError):
                dropped += 1
                continue
            if orig_id not in translation:
                dropped += 1
                continue
            new_id = translation[orig_id]
            out_lines.append(" ".join([str(new_id), *parts[1:]]))
            written += 1
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    with dst_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
        if out_lines:
            f.write("\n")
    return written, dropped


def discover_source_layout(src: Path) -> "list[tuple[Path, Path, str]]":
    """Detecta la estructura del dataset y devuelve las parejas a procesar.

    Retorna una lista de tuplas `(images_dir, labels_dir, prefix)`. El
    prefijo se antepone al nombre de cada archivo de salida para evitar
    colisiones cuando un dataset trae varios splits (Roboflow).
    """
    flat_images = src / "images"
    flat_labels = src / "labels"
    if flat_images.exists() and flat_labels.exists():
        return [(flat_images, flat_labels, "")]

    pairs: "list[tuple[Path, Path, str]]" = []
    for child in sorted(src.iterdir()):
        if not child.is_dir():
            continue
        imgs = child / "images"
        lbls = child / "labels"
        if imgs.exists() and lbls.exists():
            pairs.append((imgs, lbls, f"{child.name}__"))
    return pairs


def link_or_copy_image(src_img: Path, dst_img: Path, copy: bool) -> None:
    """Copia o enlaza una imagen al destino. Crea directorios padres."""
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    if dst_img.exists():
        return
    if copy:
        shutil.copy2(src_img, dst_img)
    else:
        # Enlace duro como atajo barato y portable; si falla (cruzar discos
        # o sistema sin soporte), caer a copia.
        try:
            dst_img.hardlink_to(src_img)
        except OSError:
            shutil.copy2(src_img, dst_img)


def main() -> int:
    """Punto de entrada principal."""
    args = parse_args()

    mapping_cfg = load_yaml(args.mapping)
    id_to_name = resolve_id_to_name(args.src, args.names_yaml)
    translation, discarded = build_id_translation(
        args.dataset, mapping_cfg, id_to_name
    )

    print(f"[remap] Dataset           : {args.dataset}")
    print(f"[remap] Fuente            : {args.src}")
    print(f"[remap] Destino           : {args.dst}")
    print(f"[remap] Tabla mapeo       : {args.mapping}")
    print(f"[remap] Clases originales : {id_to_name}")
    print(f"[remap] Traducción IDs    : {translation}")
    if discarded:
        print(f"[remap] Clases descartadas: {discarded}")

    layout = discover_source_layout(args.src)
    if not layout:
        print(
            f"[remap][error] No se detectó estructura compatible en {args.src}. "
            f"Se esperaba <src>/images + <src>/labels, o "
            f"<src>/{{train,valid,test}}/{{images,labels}}.",
            file=sys.stderr,
        )
        return 2

    print(f"[remap] Splits detectados : {[p[2].rstrip('_') or 'flat' for p in layout]}")

    dst_images = args.dst / "images"
    dst_labels = args.dst / "labels"

    total_lines = 0
    total_dropped = 0
    image_count = 0

    for src_images, src_labels, prefix in layout:
        for label_file in sorted(src_labels.rglob("*.txt")):
            relative = label_file.relative_to(src_labels)
            # Buscar la imagen pareja por nombre base (cualquier extensión soportada).
            img = None
            for ext in IMAGE_EXTS:
                candidate = src_images / relative.with_suffix(ext)
                if candidate.exists():
                    img = candidate
                    break
            if img is None:
                # Etiqueta huérfana: la ignoramos.
                continue

            out_stem = prefix + relative.stem
            dst_label_file = dst_labels / f"{out_stem}.txt"
            dst_image_file = dst_images / f"{out_stem}{img.suffix.lower()}"

            if args.dry_run:
                total_lines += 1
                continue

            written, dropped = remap_label_file(label_file, dst_label_file, translation)
            total_lines += written
            total_dropped += dropped
            link_or_copy_image(img, dst_image_file, copy=args.copy_images)
            image_count += 1

    print(
        f"[remap] OK: {image_count} imágenes, {total_lines} bboxes escritos, "
        f"{total_dropped} descartados."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
