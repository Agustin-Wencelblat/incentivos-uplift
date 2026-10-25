"""
Construye las tablas del proyecto a partir de staging (etapa 1).

    python -m src.construir

Pasos:
  1. sql/01_tipos.sql      staging (texto) -> clientes, experimento, compras,
                           transacciones. Al terminar borra stg_purchases
                           (ocupa varios GB).
  2. sql/02_variables.sql  variables por cliente y la vista datos_experimento.

Si stg_purchases ya no existe pero las tablas con tipos sí, se salta el paso 1
y solo se recalculan las variables. Sirve para iterar sobre 02_variables.sql sin
volver a cargar los CSV.
"""

from __future__ import annotations

import sys
import time

from . import db
from .config import DIR_SQL


def construir() -> None:
    if db.existe_tabla("stg_purchases"):
        t0 = time.time()
        print("Construyendo tablas con tipos (sql/01_tipos.sql)...", flush=True)
        db.ejecutar_script(DIR_SQL / "01_tipos.sql")
        print(f"  listo en {time.time() - t0:.0f}s")
    elif db.existe_tabla("transacciones"):
        print("Las tablas con tipos ya existen y stg_purchases fue borrada: se recalculan solo las variables.")
    else:
        sys.exit("No hay datos cargados. Corré primero: python -m src.carga <carpeta>")

    t0 = time.time()
    print("Calculando variables por cliente (sql/02_variables.sql)...", flush=True)
    db.ejecutar_script(DIR_SQL / "02_variables.sql")
    print(f"  listo en {time.time() - t0:.0f}s\n")

    for tabla in ["clientes", "experimento", "transacciones", "variables_clientes"]:
        print(f"  {tabla:<20} {db.contar(tabla):>12,} filas")
    fallidas = db.consulta("SELECT coalesce(sum(no_convertidos), 0) AS n FROM control_conversiones")
    n = int(fallidas["n"].iloc[0])
    if n:
        print(f"\n  AVISO: {n:,} valores no se pudieron convertir. Revisá la consulta 1 de sql/03_calidad.sql.")

    print("\nSiguientes pasos:")
    print("  - Controles de calidad: sql/03_calidad.sql en pgAdmin (una consulta a la vez)")
    print("  - Etapa 2: python -m src.estudios.experimento")


if __name__ == "__main__":
    construir()