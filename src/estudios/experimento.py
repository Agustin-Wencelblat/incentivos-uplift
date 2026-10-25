"""
Etapa 2 — El experimento:

    python -m src.estudios.experimento
    python -m src.estudios.experimento --rapido     (menos repeticiones de bootstrap)

Genera en results/etapa2/:
    balance.csv, balance.png      diferencia estandarizada (SMD) por variable
    efecto.csv                    efecto promedio: normal, bootstrap y CUPED
    segmentos.csv, segmentos.png  efecto por segmento, con p-valores de Holm
    resumen.md                    todo lo anterior en texto, para el README
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import datos, evaluacion, graficos
from ..config import VARIABLE_PRE_TRATAMIENTO, VARIABLES_MODELO
from ._comun import argumentos, carpeta, guardar_tabla, tabla_markdown

UMBRAL_SMD = 0.1


def segmentos(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Segmentos simples, definidos con variables PREVIAS al tratamiento."""
    return {
        "Frecuencia": pd.qcut(
            df["n_transacciones"].rank(method="first"), 4,
            labels=["Q1 (menos frecuentes)", "Q2", "Q3", "Q4 (más frecuentes)"],
        ).astype(str),
        "Usa el programa de puntos": pd.Series(
            np.where(df["frac_trans_con_canje"].fillna(0) > 0, "sí", "no"), index=df.index
        ),
        # Mitades por recencia (y no "compró en los últimos 30 días": en X5 casi
        # todos los clientes del experimento compraron en ese período)
        "Recencia": pd.Series(
            np.where(df["dias_desde_ultima"] <= df["dias_desde_ultima"].median(),
                     "compra más reciente que la mediana", "compra menos reciente que la mediana"),
            index=df.index,
        ),
    }


def efectos_por_segmento(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for dimension, etiquetas in segmentos(df).items():
        for valor in sorted(etiquetas.dropna().unique()):
            m = (etiquetas == valor).to_numpy()
            r = evaluacion.efecto_promedio(df.loc[m, "y"], df.loc[m, "t"])
            filas.append({"dimension": dimension, "segmento": valor, "clientes": int(m.sum()), **r})
    tabla = pd.DataFrame(filas)
    # Son varias comparaciones a la vez: sin corregir, alguna saldría
    # "significativa" por azar. Holm controla la probabilidad de al menos un
    # falso positivo en toda la familia de tests.
    tabla["p_holm"] = evaluacion.holm(tabla["p_valor"])
    return tabla[["dimension", "segmento", "clientes", "efecto", "ic_inf", "ic_sup", "p_valor", "p_holm"]]


def main(argv=None) -> None:
    args = argumentos(__doc__).parse_args(argv)
    destino = carpeta("etapa2")
    df = datos.cargar_experimento()
    X = df[VARIABLES_MODELO]
    print(f"Clientes del experimento: {len(df):,}")

    # ── 1. Balance ──────────────────────────────────────────────────────────
    print("Balance entre tratados y control...")
    balance = evaluacion.diferencia_estandarizada(X, df["t"])
    auc = evaluacion.auc_balance(X, df["t"])
    guardar_tabla(balance, destino / "balance.csv")
    graficos.barras_horizontales(
        balance.set_index("variable")["smd"], destino / "balance.png",
        "Balance entre tratados y control",
        f"Diferencia estandarizada de medias (líneas: ±{UMBRAL_SMD}, umbral de balance)",
        umbral=UMBRAL_SMD,
    )
    # Bajo asignación al azar, el SMD de una variable tiene un desvío aproximado
    # de sqrt(1/n_T + 1/n_C). Con muestras chicas, superar 0.1 puede ser puro
    # azar; con las ~200 mil filas de X5 ese ruido es ~0.0045 y la regla de 0.1
    # es la que manda. Se marca una variable si supera las dos cosas.
    n_t, n_c = int((df["t"] == 1).sum()), int((df["t"] == 0).sum())
    ruido_smd = float(np.sqrt(1 / n_t + 1 / n_c))
    umbral = max(UMBRAL_SMD, 3 * ruido_smd)
    desbalanceadas = balance.loc[balance["smd"].abs() > umbral, "variable"].tolist()

    # ── 2. Efecto promedio ──────────────────────────────────────────────────
    print("Efecto promedio (normal, bootstrap y CUPED)...")
    normal = evaluacion.efecto_promedio(df["y"], df["t"])
    boot = evaluacion.efecto_bootstrap(df["y"], df["t"], repeticiones=200 if args.rapido else 1000)
    cuped = evaluacion.efecto_cuped(df["y"], df["t"], df[VARIABLE_PRE_TRATAMIENTO])
    efecto = pd.DataFrame(
        [
            {"metodo": "Diferencia de medias, IC normal", **normal},
            {"metodo": "Diferencia de medias, IC bootstrap", **boot},
            {"metodo": f"CUPED con {VARIABLE_PRE_TRATAMIENTO}, IC normal", **cuped},
        ]
    )
    efecto["ancho_ic"] = efecto["ic_sup"] - efecto["ic_inf"]
    columnas = ["metodo", "efecto", "error_estandar", "ic_inf", "ic_sup", "ancho_ic", "p_valor"]
    guardar_tabla(efecto[[c for c in columnas if c in efecto]], destino / "efecto.csv", 5)

    # ── 3. Efecto por segmento ──────────────────────────────────────────────
    print("Efecto por segmento...")
    seg = efectos_por_segmento(df)
    guardar_tabla(seg, destino / "segmentos.csv", 5)
    graficos.efectos_con_intervalo(
        seg.assign(etiqueta=seg["dimension"] + ": " + seg["segmento"]),
        destino / "segmentos.png",
        "Efecto del incentivo por segmento",
        efecto_global=normal["efecto"],
    )

    # ── Resumen ─────────────────────────────────────────────────────────────
    reduccion_ancho = 1 - efecto["ancho_ic"].iloc[2] / efecto["ancho_ic"].iloc[0]
    if desbalanceadas or auc > 0.55:
        alerta_balance = (
            "**ATENCIÓN: hay señales de desbalance.** Antes de seguir, revisar si la asignación "
            "fue realmente al azar; si no lo fue, el efecto de esta etapa está sesgado y los "
            "métodos de las etapas siguientes tienen que tratarlo como dato observacional."
        )
    else:
        alerta_balance = "Sin señales de desbalance: compatible con una asignación al azar."

    resumen = f"""# Etapa 2 — El experimento

## ¿Fue al azar?

- Clientes: {len(df):,} ({100 * df['t'].mean():.1f}% tratados)
- Mayor |SMD|: {balance['smd'].abs().max():.4f} ({balance['variable'].iloc[0]}); variables con |SMD| > {UMBRAL_SMD}: {int((balance['smd'].abs() > UMBRAL_SMD).sum())} de {len(balance)}
- Ruido esperado del SMD por azar con este tamaño de muestra: ≈ {ruido_smd:.4f}, así que el umbral usado es max(0.1, 3 × ruido) = {umbral:.4f}
- Variables por encima del umbral: {len(desbalanceadas)}{' (' + ', '.join(desbalanceadas) + ')' if desbalanceadas else ''}
- AUC de un clasificador que intenta predecir el tratamiento: {auc:.3f} (≈ 0.5 si fue al azar)

{alerta_balance}

## ¿Hubo efecto?

- Tasa de compra: {df.loc[df['t'] == 1, 'y'].mean():.4f} en tratados, {df.loc[df['t'] == 0, 'y'].mean():.4f} en control
- Efecto promedio: {normal['efecto']:.4f}, IC 95% [{normal['ic_inf']:.4f}, {normal['ic_sup']:.4f}], p = {normal['p_valor']:.2g}
- IC bootstrap: [{boot['ic_inf']:.4f}, {boot['ic_sup']:.4f}]
- Con CUPED: {cuped['efecto']:.4f}, IC 95% [{cuped['ic_inf']:.4f}, {cuped['ic_sup']:.4f}]
  - Correlación entre el resultado y {VARIABLE_PRE_TRATAMIENTO}: ρ = {cuped['rho']:.3f}
  - Reducción de varianza esperada: ρ² = {100 * cuped['reduccion_varianza']:.1f}%; intervalo {100 * reduccion_ancho:.1f}% más angosto

## Efecto por segmento (p-valores corregidos por Holm)

{tabla_markdown(seg[['dimension', 'segmento', 'clientes', 'efecto', 'ic_inf', 'ic_sup', 'p_holm']])}

![Efecto por segmento](segmentos.png)
"""
    (destino / "resumen.md").write_text(resumen, encoding="utf-8")
    print("\n" + resumen)
    print(f"Resultados en {destino}")


if __name__ == "__main__":
    main()