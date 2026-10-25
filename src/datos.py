"""Lectura de los datos del experimento y de la población desde Postgres."""

from __future__ import annotations

import pandas as pd

from .config import VARIABLES_MODELO
from .db import consulta


def _a_float(df: pd.DataFrame) -> pd.DataFrame:
    """Postgres devuelve NUMERIC como Decimal: se pasa todo a float."""
    df = df.copy()
    for c in VARIABLES_MODELO:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df


def cargar_experimento() -> pd.DataFrame:
    """Clientes del experimento con variables, tratamiento (t) y resultado (y)."""
    df = consulta("SELECT * FROM datos_experimento ORDER BY client_id")
    df = _a_float(df).rename(columns={"tratamiento": "t", "compro": "y"})
    df["t"] = df["t"].astype(int)
    df["y"] = df["y"].astype(int)
    return df


def cargar_poblacion() -> pd.DataFrame:
    """Todos los clientes con sus variables (la población a la que se le asignan incentivos)."""
    return _a_float(consulta("SELECT * FROM variables_clientes ORDER BY client_id"))
