"""
Carga los CSV de X5 RetailHero en tablas de staging de Postgres.

    python -m src.carga data/x5                 # carga completa
    python -m src.carga data/x5 --muestra 0.2   # 20% de los clientes, para iterar rápido

La carpeta tiene que contener (comprimidos con .gz o no):
    clients.csv           información de los clientes
    uplift_train.csv      (o train.csv) tratamiento y resultado
    purchases.csv         historial de compras previo a la comunicación

Todo se carga como texto (stg_*): la conversión de tipos se hace en SQL
(sql/01_tipos.sql), donde queda documentada y se cuentan los valores que no
se pudieron convertir.

Carga completa: el archivo se manda tal cual a COPY, sin parsearlo en Python
(es lo más rápido para los ~46 millones de filas de purchases).
Con --muestra: se parsea fila por fila y se conservan los clientes cuyo hash
cae en la fracción pedida. El hash es determinístico, así que la muestra es la
misma en las tres tablas y en todas las corridas.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
import time
import zlib
from pathlib import Path

from .db import conectar

# tabla de staging -> nombres posibles del archivo (sin extensión)
ARCHIVOS = {
    "stg_clients": ["clients"],
    "stg_train": ["uplift_train", "train"],
    "stg_purchases": ["purchases"],
}

# Columnas que usa el SQL. Si falta alguna, se avisa antes de cargar.
ESPERADAS = {
    "stg_clients": {"client_id", "first_issue_date", "first_redeem_date", "age", "gender"},
    "stg_train": {"client_id", "treatment_flg", "target"},
    "stg_purchases": {
        "client_id", "transaction_id", "transaction_datetime", "regular_points_received",
        "express_points_received", "regular_points_spent", "express_points_spent",
        "purchase_sum", "store_id", "product_id", "product_quantity",
    },
}

# Algunas versiones del dataset usan otro nombre para el id del cliente
RENOMBRES = {"customer_id": "client_id"}


def buscar_archivo(carpeta: Path, nombres: list[str]) -> Path:
    for nombre in nombres:
        for extension in (".csv.gz", ".csv"):
            ruta = carpeta / f"{nombre}{extension}"
            if ruta.exists():
                return ruta
    sys.exit(f"No encontré {' ni '.join(nombres)} (.csv o .csv.gz) en {carpeta}")


def abrir_texto(ruta: Path):
    if ruta.suffix == ".gz":
        return gzip.open(ruta, "rt", encoding="utf-8", newline="")
    return open(ruta, encoding="utf-8", newline="")


def abrir_binario(ruta: Path):
    return gzip.open(ruta, "rb") if ruta.suffix == ".gz" else open(ruta, "rb")


def leer_encabezado(ruta: Path) -> list[str]:
    with abrir_texto(ruta) as f:
        encabezado = next(csv.reader(f))
    columnas = [c.strip().lower() for c in encabezado]
    return [RENOMBRES.get(c, c) for c in columnas]


def cliente_en_muestra(client_id: str, fraccion: float) -> bool:
    return zlib.crc32(client_id.encode()) % 10_000 < fraccion * 10_000


def crear_staging(con, tabla: str, columnas: list[str]) -> None:
    definicion = ", ".join(f'"{c}" TEXT' for c in columnas)
    con.execute(f"DROP TABLE IF EXISTS {tabla}")
    con.execute(f"CREATE UNLOGGED TABLE {tabla} ({definicion})")
    # UNLOGGED: no escribe el WAL. Más rápido; si la base se cae a mitad de la
    # carga, la tabla se pierde, pero es staging y se vuelve a cargar.


def cargar_completo(con, tabla: str, ruta: Path, columnas: list[str]) -> None:
    lista = ", ".join(f'"{c}"' for c in columnas)
    with abrir_binario(ruta) as f, con.cursor().copy(
        f"COPY {tabla} ({lista}) FROM STDIN WITH (FORMAT csv, HEADER true)"
    ) as copia:
        while bloque := f.read(1 << 22):
            copia.write(bloque)


def cargar_muestra(con, tabla: str, ruta: Path, columnas: list[str], fraccion: float) -> int:
    i_cliente = columnas.index("client_id")
    lista = ", ".join(f'"{c}"' for c in columnas)
    descartadas = 0
    with abrir_texto(ruta) as f, con.cursor().copy(f"COPY {tabla} ({lista}) FROM STDIN") as copia:
        lector = csv.reader(f)
        next(lector)
        for fila in lector:
            if len(fila) != len(columnas):
                descartadas += 1
                continue
            if cliente_en_muestra(fila[i_cliente], fraccion):
                copia.write_row([v if v != "" else None for v in fila])
    return descartadas


def cargar(carpeta: Path, fraccion: float | None = None) -> None:
    csv.field_size_limit(2**31 - 1)
    with conectar() as con:
        for tabla, nombres in ARCHIVOS.items():
            ruta = buscar_archivo(carpeta, nombres)
            columnas = leer_encabezado(ruta)
            faltantes = ESPERADAS[tabla] - set(columnas)
            if faltantes:
                sys.exit(
                    f"{ruta.name}: faltan columnas {sorted(faltantes)}.\n"
                    f"Columnas encontradas: {columnas}\n"
                    "Revisá la versión del dataset o adaptá sql/01_tipos.sql."
                )

            t0 = time.time()
            print(f"Cargando {ruta.name} -> {tabla} ...", flush=True)
            crear_staging(con, tabla, columnas)
            if fraccion is None:
                cargar_completo(con, tabla, ruta, columnas)
            else:
                descartadas = cargar_muestra(con, tabla, ruta, columnas, fraccion)
                if descartadas:
                    print(f"  aviso: {descartadas:,} filas mal formadas descartadas")
            con.execute(f"ANALYZE {tabla}")
            con.commit()
            n = con.execute(f"SELECT count(*) FROM {tabla}").fetchone()[0]
            print(f"  {n:,} filas en {time.time() - t0:.0f}s")

    print("\nSiguiente paso: python -m src.construir")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("carpeta", type=Path, help="carpeta con los CSV de X5")
    p.add_argument("--muestra", type=float, default=None, help="fracción de clientes a cargar (0-1)")
    a = p.parse_args()
    if a.muestra is not None and not 0 < a.muestra <= 1:
        sys.exit("--muestra tiene que estar entre 0 y 1")
    cargar(a.carpeta, a.muestra)


if __name__ == "__main__":
    main()
