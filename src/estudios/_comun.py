"""Utilidades compartidas por los estudios."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..config import DIR_RESULTADOS


def carpeta(etapa: str) -> Path:
    destino = DIR_RESULTADOS / etapa
    destino.mkdir(parents=True, exist_ok=True)
    return destino


def guardar_tabla(df: pd.DataFrame, ruta: Path, decimales: int = 4) -> None:
    df.to_csv(ruta, index=False, float_format=f"%.{decimales}f")


def argumentos(descripcion: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=descripcion, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rapido", action="store_true",
                   help="menos repeticiones y muestras más chicas (para probar que todo corre)")
    return p


def tabla_markdown(df: pd.DataFrame, decimales: int = 4) -> str:
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.{decimales}f}"
        return str(v)

    cab = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    filas = ["| " + " | ".join(fmt(v) for v in fila) + " |" for fila in df.itertuples(index=False)]
    return "\n".join([cab, sep, *filas])
