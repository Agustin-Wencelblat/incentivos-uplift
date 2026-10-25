"""Acceso a PostgreSQL: consultas a DataFrame, scripts SQL y escritura rápida con COPY."""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from sqlalchemy import create_engine, text

from .config import ConexionDB


@lru_cache(maxsize=1)
def _motor():
    return create_engine(ConexionDB.desde_entorno().url_sqlalchemy())


def conectar() -> psycopg.Connection:
    return psycopg.connect(**ConexionDB.desde_entorno().parametros_psycopg())


def hay_base() -> bool:
    """True si la base responde (los tests de integración se saltean si no)."""
    try:
        with psycopg.connect(**ConexionDB.desde_entorno().parametros_psycopg(), connect_timeout=3):
            return True
    except psycopg.OperationalError:
        return False


def consulta(sql: str, **parametros) -> pd.DataFrame:
    """SELECT -> DataFrame. Parámetros con la sintaxis :nombre."""
    with _motor().connect() as con:
        return pd.read_sql(text(sql), con, params=parametros)


def ejecutar_script(ruta: Path) -> None:
    """Ejecuta un archivo .sql completo (varias sentencias) en una transacción."""
    with conectar() as con:
        con.execute(ruta.read_text(encoding="utf-8"))


def contar(tabla: str) -> int:
    with conectar() as con:
        return con.execute(f"SELECT count(*) FROM {tabla}").fetchone()[0]


_TIPOS_PG = {"i": "BIGINT", "u": "BIGINT", "f": "DOUBLE PRECISION", "b": "BOOLEAN", "M": "TIMESTAMP"}


def escribir_tabla(df: pd.DataFrame, tabla: str, reemplazar: bool = True) -> None:
    """
    Escribe un DataFrame en Postgres con COPY (mucho más rápido que to_sql para
    cientos de miles de filas). Los tipos se infieren de los dtypes de pandas.
    """
    columnas = ", ".join(f'"{c}" {_TIPOS_PG.get(df[c].dtype.kind, "TEXT")}' for c in df.columns)
    buffer = io.StringIO()
    df.replace({np.nan: None}).to_csv(buffer, index=False, header=False, na_rep="")
    buffer.seek(0)
    with conectar() as con:
        if reemplazar:
            con.execute(f"DROP TABLE IF EXISTS {tabla}")
        con.execute(f"CREATE TABLE IF NOT EXISTS {tabla} ({columnas})")
        with con.cursor().copy(f"COPY {tabla} FROM STDIN WITH (FORMAT csv)") as copia:
            while bloque := buffer.read(1 << 20):
                copia.write(bloque)
