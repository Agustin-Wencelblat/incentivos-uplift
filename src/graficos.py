"""Estilo común y gráficos del proyecto (matplotlib, sin interfaz)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paleta categórica en orden fijo (validada para daltonismo) y tintas de texto
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TINTA, TINTA_2, GRILLA = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.edgecolor": GRILLA,
        "axes.labelcolor": TINTA_2,
        "axes.titlecolor": TINTA,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRILLA,
        "grid.linewidth": 0.6,
        "xtick.color": TINTA_2,
        "ytick.color": TINTA_2,
        "legend.frameon": False,
        "lines.linewidth": 2,
    }
)


def guardar(fig, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta)
    plt.close(fig)


def barras_horizontales(valores: pd.Series, ruta: Path, titulo: str, etiqueta_x: str,
                        umbral: float | None = None) -> None:
    """Barras horizontales; con umbral, marca ±umbral y lo deja siempre a la vista."""
    fig, ax = plt.subplots(figsize=(6, 0.32 * len(valores) + 1.2))
    ax.set_axisbelow(True)
    y = np.arange(len(valores))[::-1]
    ax.barh(y, valores, color=COLORES[0], height=0.6)
    ax.axvline(0, color=TINTA_2, lw=1)
    if umbral is not None:
        for signo in (-1, 1):
            ax.axvline(signo * umbral, color=COLORES[1], lw=1.2, ls="--")
        limite = max(1.2 * umbral, 1.1 * float(np.abs(valores).max()))
        ax.set_xlim(-limite, limite)
    ax.set_yticks(y, valores.index, color=TINTA)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(etiqueta_x)
    ax.set_title(titulo)
    guardar(fig, ruta)


def efectos_con_intervalo(tabla: pd.DataFrame, ruta: Path, titulo: str, efecto_global: float | None = None) -> None:
    """
    Gráfico de efectos con intervalo (forest plot). tabla: columnas etiqueta,
    efecto, ic_inf, ic_sup, en el orden en que se quieren mostrar (arriba a abajo).
    """
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(tabla) + 1.3))
    y = np.arange(len(tabla))[::-1]
    ax.errorbar(
        tabla["efecto"], y,
        xerr=[tabla["efecto"] - tabla["ic_inf"], tabla["ic_sup"] - tabla["efecto"]],
        fmt="o", ms=5, color=COLORES[0], ecolor=COLORES[0], elinewidth=2, capsize=0,
    )
    ax.axvline(0, color=TINTA_2, lw=1)
    if efecto_global is not None:
        ax.axvline(efecto_global, color=COLORES[1], lw=1.2, ls="--", label="Efecto promedio global")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8)
    ax.set_yticks(y, tabla["etiqueta"], color=TINTA)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Efecto en la probabilidad de compra (tratados − control), con IC 95%")
    ax.set_title(titulo)
    guardar(fig, ruta)
