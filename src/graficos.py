"""Estilo común y gráficos del proyecto (matplotlib, sin interfaz)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

COLORES = ["#2a78d6", "#b3fdbf", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
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


def efectos_con_intervalo(tabla: pd.DataFrame, ruta: Path, titulo: str, efecto_global: float | None = None,
                          etiqueta_x: str = "Efecto en la probabilidad de compra (tratados − control), con IC 95%",
                          etiqueta_global: str = "Efecto promedio global") -> None:
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
        ax.axvline(efecto_global, color=COLORES[1], lw=1.2, ls="--", label=etiqueta_global)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8)
    ax.set_yticks(y, tabla["etiqueta"], color=TINTA)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(etiqueta_x)
    ax.set_title(titulo)
    guardar(fig, ruta)


def puntos_por_grupo(tabla: pd.DataFrame, ruta: Path, titulo: str, etiqueta_x: str,
                     referencia: float | None = None) -> None:
    """
    Una fila por grupo (p. ej. escenario) y un punto con intervalo por serie
    (p. ej. método). tabla: columnas grupo, serie, media, ic_inf, ic_sup; los
    grupos y las series se muestran en el orden en que aparecen.
    """
    grupos = list(dict.fromkeys(tabla["grupo"]))
    series = list(dict.fromkeys(tabla["serie"]))
    fig, ax = plt.subplots(figsize=(6.6, 0.55 * len(grupos) + 1.6))
    ancho = 0.7 / max(len(series), 1)
    base = np.arange(len(grupos))[::-1]
    for i, serie in enumerate(series):
        d = tabla[tabla["serie"] == serie].set_index("grupo").reindex(grupos)
        y = base + 0.35 - ancho / 2 - i * ancho
        ax.errorbar(
            d["media"], y, xerr=[d["media"] - d["ic_inf"], d["ic_sup"] - d["media"]],
            fmt="o", ms=4, color=COLORES[i % len(COLORES)], elinewidth=1.5, capsize=0, label=serie,
        )
    if referencia is not None:
        ax.axvline(referencia, color=TINTA_2, lw=1, ls="--")
    ax.set_yticks(base, grupos, color=TINTA)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(etiqueta_x)
    ax.set_title(titulo)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.6 / (len(grupos) + 2)),
              ncol=min(len(series), 3), fontsize=8)
    guardar(fig, ruta)


def curvas_qini(curvas: dict[str, pd.DataFrame], ruta: Path, titulo: str) -> None:
    """curvas: nombre -> DataFrame con fraccion e incremental (ver evaluacion.curva_qini)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    total, i = None, 0
    for nombre, c in curvas.items():
        if nombre.lower().startswith("oráculo"):
            estilo = {"color": TINTA_2, "ls": ":"}
        else:
            estilo, i = {"color": COLORES[(i + 1) % len(COLORES)]}, i + 1   # COLORES[0] es "sin modelo"
        ax.plot(c["fraccion"], c["incremental"], label=nombre, lw=1.6, **estilo)
        total = c["incremental"].iloc[-1]
    if total is not None:
        ax.plot([0, 1], [0, total], color=TINTA_2, lw=1, ls="--", label="Azar")
    ax.axhline(0, color=GRILLA, lw=1)
    ax.set_xlabel("Fracción de clientes tratados (de mayor a menor uplift predicho)")
    ax.set_ylabel("Compras incrementales por cliente")
    ax.set_title(titulo)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=8)
    guardar(fig, ruta)


def calibracion(tabla: pd.DataFrame, ruta: Path, titulo: str) -> None:
    """tabla: salida de evaluacion.calibracion_por_grupo."""
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.errorbar(
        tabla["uplift_predicho"], tabla["efecto_observado"],
        yerr=[tabla["efecto_observado"] - tabla["ic_inf"], tabla["ic_sup"] - tabla["efecto_observado"]],
        fmt="o", ms=5, color=COLORES[0], elinewidth=1.5, capsize=0,
    )
    for _, f in tabla.iterrows():
        ax.annotate(str(int(f["grupo"])), (f["uplift_predicho"], f["efecto_observado"]),
                    xytext=(5, 3), textcoords="offset points", fontsize=7, color=TINTA_2)
    bajo = min(tabla["uplift_predicho"].min(), tabla["ic_inf"].min())
    alto = max(tabla["uplift_predicho"].max(), tabla["ic_sup"].max())
    ax.plot([bajo, alto], [bajo, alto], color=TINTA_2, lw=1, ls="--", label="Calibración perfecta")
    ax.axhline(0, color=GRILLA, lw=1)
    ax.set_xlabel("Uplift predicho medio del grupo")
    ax.set_ylabel("Efecto observado (tratados − control), IC 95%")
    ax.set_title(titulo)
    ax.legend(fontsize=8, loc="upper left")
    guardar(fig, ruta)


def lineas(tabla: pd.DataFrame, x: str, ruta: Path, titulo: str, etiqueta_x: str, etiqueta_y: str,
           banda: bool = True, referencia_y: float | None = None) -> None:
    """
    Una línea por serie. tabla: columnas serie, x, valor y (si banda) ic_inf,
    ic_sup. Las series se dibujan en el orden en que aparecen.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, serie in enumerate(dict.fromkeys(tabla["serie"])):
        d = tabla[tabla["serie"] == serie].sort_values(x)
        color = COLORES[i % len(COLORES)]
        ax.plot(d[x], d["valor"], color=color, label=serie, marker="o", ms=3)
        if banda and "ic_inf" in d:
            ax.fill_between(d[x], d["ic_inf"], d["ic_sup"], color=color, alpha=0.15, lw=0)
    if referencia_y is not None:
        ax.axhline(referencia_y, color=TINTA_2, lw=1, ls="--")
    ax.set_xlabel(etiqueta_x)
    ax.set_ylabel(etiqueta_y)
    ax.set_title(titulo)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8)
    guardar(fig, ruta)


def mapa_calor(tabla: pd.DataFrame, ruta: Path, titulo: str, etiqueta: str, tope: float | None = None) -> None:
    """tabla: filas y columnas con nombre, valores numéricos. El color se satura en `tope`."""
    fig, ax = plt.subplots(figsize=(1.1 * tabla.shape[1] + 2.6, 0.3 * tabla.shape[0] + 1.4))
    tope = tope if tope is not None else float(np.nanmax(tabla.to_numpy()))
    mapa = LinearSegmentedColormap.from_list("secuencial", ["#f4f3ef", COLORES[0], "#0d2f5c"])
    im = ax.imshow(np.clip(tabla.to_numpy(dtype=float), 0, tope), cmap=mapa, vmin=0, vmax=tope, aspect="auto")
    for i in range(tabla.shape[0]):
        for j in range(tabla.shape[1]):
            val = tabla.iat[i, j]
            if np.isfinite(val):
                color = "white" if val > 0.55 * tope else TINTA
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
    ax.set_xticks(range(tabla.shape[1]), tabla.columns, rotation=30, ha="right", color=TINTA)
    ax.set_yticks(range(tabla.shape[0]), tabla.index, color=TINTA)
    ax.grid(False)
    for lado in ax.spines.values():
        lado.set_visible(False)
    barra = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    barra.set_label(etiqueta, color=TINTA_2)
    barra.outline.set_visible(False)
    ax.set_title(titulo)
    guardar(fig, ruta)