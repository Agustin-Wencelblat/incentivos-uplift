"""
Etapa 6: validación de datos y monitoreo de drift.

Validación (antes de puntuar): los datos nuevos tienen que parecerse en forma a
los de entrenamiento: mismas columnas, numéricas, sin una suba brusca de
faltantes, sin clientes duplicados. Si algo falla, el pipeline frena.

Drift (después): ¿cambió la distribución de las variables o de las
predicciones? Dos medidas por variable:

    PSI = sum_b (a_b - r_b) ln(a_b / r_b)

con r_b y a_b la proporción de clientes en el bin b en la referencia y en los
datos actuales (bins = deciles de la referencia, más uno para faltantes). Es la
medida habitual en riesgo; reglas usuales: < 0.1 estable, 0.1-0.25 revisar,
> 0.25 cambio importante. Ojo: con muestras chicas el PSI es positivo aunque no
haya drift; su valor esperado sin drift es ~ (bins - 1)(1/n_ref + 1/n_act).

Wasserstein-1 sobre la variable estandarizada con la referencia: cuánta "masa"
hay que mover y cuánto, en desvíos. No depende de bins y ve cambios en las colas
que el PSI mezcla en un solo bin. La versión multivariada (sliced) promedia la
distancia sobre proyecciones al azar: detecta cambios en la relación entre
variables que ninguna marginal muestra.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from .config import SEMILLA

UMBRAL_REVISAR, UMBRAL_ALERTA = 0.10, 0.25
EPS = 1e-4


# ─── Validación ──────────────────────────────────────────────────────────────
def perfil(df: pd.DataFrame, variables: list[str]) -> dict:
    """Lo que se guarda de los datos de entrenamiento para validar los nuevos."""
    return {
        v: {
            "frac_nulos": float(df[v].isna().mean()),
            "p001": float(df[v].quantile(0.001)) if df[v].notna().any() else None,
            "p999": float(df[v].quantile(0.999)) if df[v].notna().any() else None,
        }
        for v in variables
    }


def validar(df: pd.DataFrame, perfil_ref: dict, id_columna: str = "client_id",
            max_suba_nulos: float = 0.10, max_fuera_de_rango: float = 0.01) -> pd.DataFrame:
    """
    Devuelve una tabla de problemas (nivel 'error' o 'aviso'). Con algún error,
    no hay que puntuar.
    """
    problemas = []

    def agregar(nivel, variable, mensaje):
        problemas.append({"nivel": nivel, "variable": variable, "mensaje": mensaje})

    if len(df) == 0:
        agregar("error", "-", "no hay filas")
    if id_columna in df and df[id_columna].duplicated().any():
        agregar("error", id_columna, f"{int(df[id_columna].duplicated().sum())} clientes duplicados")
    for v, ref in perfil_ref.items():
        if v not in df:
            agregar("error", v, "falta la columna")
            continue
        if not pd.api.types.is_numeric_dtype(df[v]):
            agregar("error", v, f"no es numérica ({df[v].dtype})")
            continue
        nulos = float(df[v].isna().mean())
        if nulos > ref["frac_nulos"] + max_suba_nulos:
            agregar("error", v, f"faltantes {nulos:.1%} (en entrenamiento {ref['frac_nulos']:.1%})")
        if ref["p001"] is not None and len(df):
            # Margen del 10% del rango para no marcar valores apenas fuera del entrenamiento
            margen = 0.1 * (ref["p999"] - ref["p001"])
            fuera = ((df[v] < ref["p001"] - margen) | (df[v] > ref["p999"] + margen)).mean()
            if fuera > max_fuera_de_rango:
                agregar("aviso", v, f"{fuera:.1%} de valores fuera del rango de entrenamiento")
    return pd.DataFrame(problemas, columns=["nivel", "variable", "mensaje"])


# ─── Drift ───────────────────────────────────────────────────────────────────
def psi(referencia, actual, bins: int = 10) -> float:
    r, a = np.asarray(referencia, dtype=float), np.asarray(actual, dtype=float)
    r_ok, a_ok = r[np.isfinite(r)], a[np.isfinite(a)]
    if len(r_ok):
        bordes = np.unique(np.quantile(r_ok, np.linspace(0, 1, bins + 1)[1:-1]))
        cuenta_r = np.bincount(np.searchsorted(bordes, r_ok, side="right"), minlength=len(bordes) + 1)
        cuenta_a = np.bincount(np.searchsorted(bordes, a_ok, side="right"), minlength=len(bordes) + 1)
    else:
        cuenta_r = cuenta_a = np.zeros(1)
    # Un bin más para los faltantes
    pr = np.append(cuenta_r, len(r) - len(r_ok)) / len(r)
    pa = np.append(cuenta_a, len(a) - len(a_ok)) / len(a)
    pr, pa = np.clip(pr, EPS, None), np.clip(pa, EPS, None)
    return float(np.sum((pa - pr) * np.log(pa / pr)))


def psi_esperado_sin_drift(n_ref: int, n_act: int, bins: int = 10) -> float:
    """Aproximación del PSI medio entre dos muestras de la MISMA distribución."""
    return (bins - 1) * (1 / n_ref + 1 / n_act)


def wasserstein_estandarizado(referencia, actual) -> float:
    r, a = np.asarray(referencia, dtype=float), np.asarray(actual, dtype=float)
    r, a = r[np.isfinite(r)], a[np.isfinite(a)]
    if len(r) == 0 or len(a) == 0:
        return float("nan")
    escala = r.std() or 1.0
    return float(wasserstein_distance((r - r.mean()) / escala, (a - r.mean()) / escala))


def _estandarizar_con(ref: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
    media, desvio, mediana = ref.mean(), ref.std().replace(0, 1).fillna(1), ref.median()
    return ((df.fillna(mediana) - media) / desvio).fillna(0).to_numpy()


def sliced_wasserstein(referencia: pd.DataFrame, actual: pd.DataFrame, proyecciones: int = 100,
                       semilla: int = SEMILLA) -> float:
    """Wasserstein promedio sobre direcciones al azar (variables estandarizadas con la referencia)."""
    R, A = _estandarizar_con(referencia, referencia), _estandarizar_con(referencia, actual)
    rng = np.random.default_rng(semilla)
    direcciones = rng.normal(size=(proyecciones, R.shape[1]))
    direcciones /= np.linalg.norm(direcciones, axis=1, keepdims=True)
    return float(np.mean([wasserstein_distance(R @ u, A @ u) for u in direcciones]))


def nivel_psi(valor: float) -> str:
    return "alerta" if valor > UMBRAL_ALERTA else "revisar" if valor > UMBRAL_REVISAR else "ok"


def reporte_drift(referencia: pd.DataFrame, actual: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    filas = [{
        "variable": v,
        "psi": psi(referencia[v], actual[v]),
        "wasserstein": wasserstein_estandarizado(referencia[v], actual[v]),
        "nulos_ref": float(referencia[v].isna().mean()),
        "nulos_act": float(actual[v].isna().mean()),
    } for v in variables]
    tabla = pd.DataFrame(filas)
    tabla["nivel"] = tabla["psi"].map(nivel_psi)
    return tabla.sort_values("psi", ascending=False, ignore_index=True)