"""Construye splits estratificados train/val/test sin fuga entre splits.

Toma como entrada uno o más datasets ya remapeados (típicamente
`data/interim/<dataset>/`) y produce `data/processed/` en formato Ultralytics:

    data/processed/
        images/train/  labels/train/
        images/val/    labels/val/
        images/test/   labels/test/

Estratificación: cada imagen se asigna a un "bucket" según el subconjunto
de clases presentes en su archivo de etiquetas; cada bucket se reparte por
separado en las proporciones indicadas. Esto preserva la distribución de
clases incluso cuando hay desbalance fuerte.

Sin fuga: el reparto se hace por imagen (no por bbox); una imagen pertenece
exactamente a un split. No se usan augmentaciones aquí (eso es trabajo de
Ultralytics durante el entrenamiento).
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Genera splits train/val/test estratificados por combinación de "
            "clases presentes en cada imagen."
        )
    )
    parser.add_argument(
        "--src",
        required=True,
        type=Path,
        help=(
            "Carpeta con datasets remapeados. Puede ser un dataset suelto "
            "(con images/ y labels/) o un directorio que contenga varios "
            "subdirectorios de datasets."
        ),
    )
    parser.add_argument(
        "--dst",
        required=True,
        type=Path,
        help="Carpeta destino con la estructura processed/ para Ultralytics.",
    )
    parser.add_argument(
        "--ratios",
        nargs=3,
        type=float,
        default=[0.70, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Proporciones train/val/test. Deben sumar 1.0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copia archivos en lugar de enlazarlos.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No escribe nada; solo reporta los tamaños de cada split.",
    )
    return parser.parse_args()


def discover_datasets(src: Path) -> List[Path]:
    """Devuelve la lista de raíces de dataset bajo `src`.

    Si `src` ya contiene `images/` y `labels/` directamente, se trata como
    un único dataset. En caso contrario se buscan subdirectorios con esa
    estructura.
    """
    if (src / "images").exists() and (src / "labels").exists():
        return [src]
    found = []
    for child in sorted(src.iterdir()):
        if child.is_dir() and (child / "images").exists() and (child / "labels").exists():
            found.append(child)
    return found


def read_classes_in_label(label_file: Path) -> FrozenSet[int]:
    """Lee un archivo de etiquetas YOLO y devuelve el conjunto de class_ids."""
    classes = set()
    if not label_file.exists():
        return frozenset()
    with label_file.open("r", encoding="utf-8") as f:
        for raw in f:
            parts = raw.strip().split()
            if not parts:
                continue
            try:
                classes.add(int(parts[0]))
            except ValueError:
                continue
    return frozenset(classes)


def find_image_for(label_file: Path, images_root: Path, labels_root: Path) -> Path:
    """Encuentra la imagen correspondiente a una etiqueta. Lanza si no existe."""
    relative = label_file.relative_to(labels_root)
    for ext in IMAGE_EXTS:
        candidate = images_root / relative.with_suffix(ext)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No se encontró imagen pareja para {label_file} bajo {images_root}."
    )


def stratified_split(
    items: List[Tuple[Path, Path, FrozenSet[int]]],
    ratios: Tuple[float, float, float],
    rng: random.Random,
) -> Dict[str, List[Tuple[Path, Path]]]:
    """Reparte ítems en train/val/test estratificando por combinación de clases.

    `items` es una lista de tuplas (imagen, etiqueta, clases_presentes).
    """
    train_r, val_r, _ = ratios

    buckets: Dict[FrozenSet[int], List[Tuple[Path, Path]]] = defaultdict(list)
    for img, lbl, classes in items:
        buckets[classes].append((img, lbl))

    splits: Dict[str, List[Tuple[Path, Path]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for combo, pairs in buckets.items():
        rng.shuffle(pairs)
        n = len(pairs)
        n_train = int(round(n * train_r))
        n_val = int(round(n * val_r))
        # El resto va a test (evita off-by-one por redondeos).
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        splits["train"].extend(pairs[:n_train])
        splits["val"].extend(pairs[n_train : n_train + n_val])
        splits["test"].extend(pairs[n_train + n_val :])

    return splits


def materialize(
    splits: Dict[str, List[Tuple[Path, Path]]],
    dst: Path,
    copy: bool,
) -> None:
    """Copia o enlaza imágenes y etiquetas a la estructura processed/."""
    for split_name, pairs in splits.items():
        img_out = dst / "images" / split_name
        lbl_out = dst / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img, lbl in pairs:
            # Para evitar colisiones de nombres entre datasets, prefijar con
            # la ruta relativa convertida a guiones bajos.
            stem_key = "__".join(img.with_suffix("").parts[-3:])
            img_dst = img_out / f"{stem_key}{img.suffix.lower()}"
            lbl_dst = lbl_out / f"{stem_key}.txt"
            _put(img, img_dst, copy)
            _put(lbl, lbl_dst, copy)


def _put(src_file: Path, dst_file: Path, copy: bool) -> None:
    """Coloca un archivo en destino vía copia o enlace duro."""
    if dst_file.exists():
        return
    if copy:
        shutil.copy2(src_file, dst_file)
        return
    try:
        dst_file.hardlink_to(src_file)
    except OSError:
        shutil.copy2(src_file, dst_file)


def main() -> int:
    """Punto de entrada principal."""
    args = parse_args()
    if abs(sum(args.ratios) - 1.0) > 1e-6:
        print(f"[splits][error] Los ratios deben sumar 1.0 (recibido: {sum(args.ratios)}).", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    datasets = discover_datasets(args.src)
    if not datasets:
        print(f"[splits][error] No se encontraron datasets en {args.src}.", file=sys.stderr)
        return 2

    print(f"[splits] Datasets detectados: {[d.name for d in datasets]}")

    items: List[Tuple[Path, Path, FrozenSet[int]]] = []
    for dset in datasets:
        images_root = dset / "images"
        labels_root = dset / "labels"
        for lbl in sorted(labels_root.rglob("*.txt")):
            try:
                img = find_image_for(lbl, images_root, labels_root)
            except FileNotFoundError:
                continue
            classes = read_classes_in_label(lbl)
            items.append((img, lbl, classes))

    if not items:
        print("[splits][error] No se encontraron pares imagen/etiqueta.", file=sys.stderr)
        return 2

    print(f"[splits] Total ítems con etiqueta: {len(items)}")

    splits = stratified_split(items, tuple(args.ratios), rng)
    for name, pairs in splits.items():
        print(f"[splits]   {name:5s}: {len(pairs)}")

    if args.dry_run:
        return 0

    args.dst.mkdir(parents=True, exist_ok=True)
    materialize(splits, args.dst, copy=args.copy)
    print(f"[splits] OK: estructura escrita en {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
