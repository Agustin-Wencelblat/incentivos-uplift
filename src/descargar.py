"""
Descarga los archivos de X5 RetailHero a data/x5/ y verifica su integridad.

    python -m src.descargar

Son los mismos archivos que usa scikit-uplift (sklift.datasets.fetch_x5), bajados directamente.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from .config import DIR_DATOS

BASE = "https://sklift.s3.eu-west-2.amazonaws.com"
ARCHIVOS = {
    # nombre: hash MD5 publicado por scikit-uplift
    "uplift_train.csv.gz": "2720bbb659daa9e0989b2777b6a42d19",
    "clients.csv.gz": "b9cdeb2806b732771de03e819b3354c5",
    "purchases.csv.gz": "48d2de13428e24e8b61d66fef02957a8",
}


def md5(ruta: Path) -> str:
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        while bloque := f.read(1 << 20):
            h.update(bloque)
    return h.hexdigest()


def descargar(nombre: str, destino: Path) -> None:
    url = f"{BASE}/{nombre}"
    temporal = destino.with_suffix(destino.suffix + ".parcial")
    with urllib.request.urlopen(url) as resp, open(temporal, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        bajado = 0
        while bloque := resp.read(1 << 20):
            f.write(bloque)
            bajado += len(bloque)
            if total:
                print(f"\r  {nombre}: {bajado / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)
    print()
    temporal.replace(destino)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--carpeta", type=Path, default=DIR_DATOS / "x5")
    a = p.parse_args()
    a.carpeta.mkdir(parents=True, exist_ok=True)

    for nombre, hash_esperado in ARCHIVOS.items():
        destino = a.carpeta / nombre
        if destino.exists() and md5(destino) == hash_esperado:
            print(f"  {nombre}: ya estaba descargado")
            continue
        descargar(nombre, destino)
        if md5(destino) != hash_esperado:
            sys.exit(f"{nombre}: el hash no coincide; el archivo está corrupto. Borralo y volvé a correr.")
        print(f"  {nombre}: ok")

    print(f"\nListo. Siguiente paso: python -m src.carga {a.carpeta}")


if __name__ == "__main__":
    main()
