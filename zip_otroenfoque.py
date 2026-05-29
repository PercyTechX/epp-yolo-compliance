"""Empaqueta data/processed_otroenfoque/ en un zip para subir a Kaggle.

Estructura del zip resultante (la raíz del zip ES la raíz del Kaggle Dataset):
    images/train/...
    images/val/...
    labels/train/...
    labels/val/...

Modo `ZIP_STORED`: sin compresión. Las imágenes JPEG ya están comprimidas.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC = REPO_ROOT / "data" / "processed_otroenfoque"
OUT = REPO_ROOT / "data" / "processed_otroenfoque.zip"


def main() -> int:
    """Empaqueta data/processed_otroenfoque en data/processed_otroenfoque.zip."""
    if not SRC.exists():
        print(f"[zip][error] No existe {SRC}", file=sys.stderr)
        return 2

    files = [f for f in SRC.rglob("*") if f.is_file()]
    total = len(files)
    print(f"[zip] Archivos a empaquetar : {total}")
    print(f"[zip] Destino               : {OUT}")

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for i, f in enumerate(files, 1):
            zf.write(f, f.relative_to(SRC))
            if i % 1000 == 0 or i == total:
                print(f"[zip]   {i}/{total} archivos")

    size_mb = OUT.stat().st_size / (1024 * 1024)
    print(f"[zip] OK: {OUT} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
