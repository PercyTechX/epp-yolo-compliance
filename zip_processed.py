"""Empaqueta data/processed/ en un solo zip para subir a Kaggle Dataset.

Estructura del zip resultante (la raíz del zip ES la raíz del Kaggle Dataset):
    images/train/...
    images/val/...
    images/test/...
    labels/train/...
    labels/val/...
    labels/test/...

Modo `ZIP_STORED`: sin compresión. Las imágenes JPEG ya están comprimidas;
intentar comprimirlas de nuevo gasta CPU sin reducir tamaño. Para los
.txt de etiquetas la diferencia es despreciable a esta escala.

Se ejecuta una sola vez. No es parte del pipeline; está en la raíz del repo
para que el usuario lo descubra fácil, pero no se versiona (queda en
.gitignore vía el patrón general de archivos sueltos de utilería).
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC = REPO_ROOT / "data" / "processed"
OUT = REPO_ROOT / "data" / "processed.zip"


def main() -> int:
    """Empaqueta data/processed en data/processed.zip."""
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
            if i % 5000 == 0 or i == total:
                print(f"[zip]   {i}/{total} archivos")

    size_mb = OUT.stat().st_size / (1024 * 1024)
    print(f"[zip] OK: {OUT} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
