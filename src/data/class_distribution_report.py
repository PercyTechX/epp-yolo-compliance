"""Reporta la distribución de clases por split en CSV + figura PNG.

Lee la estructura `data/processed/labels/{train,val,test}/*.txt`, cuenta
instancias por clase en cada split y genera:

  - Un CSV con la tabla cruzada split x clase.
  - Una figura de barras agrupadas (Matplotlib) lista para el informe.

Es un sanity check de la fase 1: si una clase minoritaria aparece muy
poco en train o casi nada en val/test, lo detectamos antes de entrenar.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING = REPO_ROOT / "configs" / "class_mapping.yaml"

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Reporta distribución de clases por split (CSV + figura)."
    )
    parser.add_argument(
        "--src",
        required=True,
        type=Path,
        help="Raíz del dataset procesado (contiene labels/{train,val,test}).",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Ruta del CSV destino. La figura se escribe junto al CSV (.png).",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_MAPPING,
        help="class_mapping.yaml para nombres legibles.",
    )
    return parser.parse_args()


def load_class_names(mapping_path: Path) -> Dict[int, str]:
    """Carga el mapa id -> nombre de la taxonomía unificada."""
    if not mapping_path.exists():
        return {}
    with mapping_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    unified = cfg.get("unified_classes", {})
    return {int(k): v for k, v in unified.items()}


def count_split(labels_dir: Path) -> Dict[int, int]:
    """Cuenta instancias por id de clase en un split."""
    counts: Dict[int, int] = defaultdict(int)
    if not labels_dir.exists():
        return counts
    for lbl in labels_dir.rglob("*.txt"):
        with lbl.open("r", encoding="utf-8") as f:
            for raw in f:
                parts = raw.strip().split()
                if not parts:
                    continue
                try:
                    counts[int(parts[0])] += 1
                except ValueError:
                    continue
    return counts


def write_csv(
    out_path: Path,
    counts_by_split: Dict[str, Dict[int, int]],
    names: Dict[int, str],
) -> None:
    """Escribe la tabla cruzada split x clase."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_ids = sorted({cid for c in counts_by_split.values() for cid in c} | set(names))
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["class_id", "class_name", *SPLITS, "total"]
        writer.writerow(header)
        for cid in all_ids:
            row = [
                cid,
                names.get(cid, ""),
                *(counts_by_split[s].get(cid, 0) for s in SPLITS),
            ]
            row.append(sum(counts_by_split[s].get(cid, 0) for s in SPLITS))
            writer.writerow(row)


def plot_distribution(
    fig_path: Path,
    counts_by_split: Dict[str, Dict[int, int]],
    names: Dict[int, str],
) -> None:
    """Genera figura de barras agrupadas por clase y split."""
    all_ids = sorted({cid for c in counts_by_split.values() for cid in c} | set(names))
    labels = [names.get(cid, str(cid)) for cid in all_ids]
    x = range(len(all_ids))
    width = 0.27

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, split in enumerate(SPLITS):
        values = [counts_by_split[split].get(cid, 0) for cid in all_ids]
        ax.bar([xi + (i - 1) * width for xi in x], values, width=width, label=split)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Instancias")
    ax.set_title("Distribución de clases por split")
    ax.legend()
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def main() -> int:
    """Punto de entrada principal."""
    args = parse_args()
    labels_root = args.src / "labels"
    if not labels_root.exists():
        print(f"[dist][error] No existe {labels_root}.", file=sys.stderr)
        return 2

    names = load_class_names(args.mapping)
    counts_by_split = {s: count_split(labels_root / s) for s in SPLITS}

    write_csv(args.out, counts_by_split, names)
    fig_path = args.out.with_suffix(".png")
    plot_distribution(fig_path, counts_by_split, names)

    print(f"[dist] CSV escrito en   : {args.out}")
    print(f"[dist] Figura escrita en: {fig_path}")
    for s in SPLITS:
        total = sum(counts_by_split[s].values())
        print(f"[dist]   {s:5s}: {total} bboxes en total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
