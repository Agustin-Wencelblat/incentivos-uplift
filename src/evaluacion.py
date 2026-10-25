"""
Evaluación del experimento (etapa 2): balance entre grupos, efecto promedio,
reducción de varianza con CUPED, bootstrap y corrección por comparaciones
múltiples.

Evaluación de modelos de uplift (etapa 3): curva y coeficiente Qini, PEHE y
captura de la política.
"""

# TODO (etapa 4): calibración del uplift por decil (efecto predicho vs. observado).

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .config import SEMILLA


# ═══ Experimento ═════════════════════════════════════════════════════════════
def diferencia_estandarizada(X: pd.DataFrame, t) -> pd.DataFrame:
    """
    SMD = (media_T - media_C) / sqrt((s_T^2 + s_C^2) / 2) 
    """
    t = np.asarray(t)
    filas = []
    for c in X.columns:
        x = X[c].to_numpy(dtype=float)
        x1, x0 = x[t == 1], x[t == 0]
        s = np.sqrt((np.nanvar(x1, ddof=1) + np.nanvar(x0, ddof=1)) / 2)
        smd = (np.nanmean(x1) - np.nanmean(x0)) / s if s > 0 else 0.0
        filas.append(
            {
                "variable": c,
                "media_tratados": np.nanmean(x1),
                "media_control": np.nanmean(x0),
                "smd": smd,
                "dif_pct_faltantes": 100 * (np.isnan(x1).mean() - np.isnan(x0).mean()),
            }
        )
    return pd.DataFrame(filas).sort_values("smd", key=np.abs, ascending=False, ignore_index=True)


def auc_balance(X: pd.DataFrame, t, semilla: int = SEMILLA) -> float:
    modelo = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=semilla)
    cv = StratifiedKFold(5, shuffle=True, random_state=semilla)
    p = cross_val_predict(modelo, X, t, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(t, p))


def efecto_promedio(y, t, nivel: float = 0.95) -> dict:
    """Diferencia de medias con intervalo normal (varianzas no agrupadas)."""
    y, t = np.asarray(y, dtype=float), np.asarray(t)
    y1, y0 = y[t == 1], y[t == 0]
    efecto = y1.mean() - y0.mean()
    ee = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    z = stats.norm.ppf(0.5 + nivel / 2)
    return {
        "efecto": efecto,
        "error_estandar": ee,
        "ic_inf": efecto - z * ee,
        "ic_sup": efecto + z * ee,
        "p_valor": 2 * stats.norm.sf(abs(efecto / ee)),
        "n_tratados": len(y1),
        "n_control": len(y0),
    }


def efecto_cuped(y, t, x_pre, nivel: float = 0.95) -> dict:
    """
    CUPED (Deng et al., 2013): Y_cv = Y - theta (X_pre - media(X_pre)), con
    theta = Cov(Y, X_pre) / Var(X_pre). El efecto estimado sigue siendo
    insesgado (X_pre es previa al tratamiento, así que tiene la misma media en
    los dos grupos) y la varianza se reduce en un factor 1 - rho^2.

    X_pre TIENE que ser previa al tratamiento: si el incentivo la afecta, el
    ajuste sesga el efecto.
    """
    y, x = np.asarray(y, dtype=float), np.asarray(x_pre, dtype=float)
    x = np.where(np.isnan(x), np.nanmean(x), x)
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    y_cv = y - theta * (x - x.mean())
    resultado = efecto_promedio(y_cv, t, nivel)
    rho = np.corrcoef(y, x)[0, 1]
    resultado.update({"theta": theta, "rho": rho, "reduccion_varianza": rho**2})
    return resultado


def holm(p_valores) -> np.ndarray:
    """Corrección de Holm-Bonferroni: p-valores ajustados (controla el error familiar)."""
    p = np.asarray(p_valores, dtype=float)
    m = len(p)
    orden = np.argsort(p)
    ajustados = np.empty(m)
    acumulado = 0.0
    for rango, i in enumerate(orden):
        acumulado = max(acumulado, (m - rango) * p[i])
        ajustados[i] = min(1.0, acumulado)
    return ajustados


# ═══ Bootstrap ═══════════════════════════════════════════════════════════════
def bootstrap(
    estadistico: Callable[..., float],
    *arreglos,
    repeticiones: int = 500,
    nivel: float = 0.95,
    semilla: int = SEMILLA,
) -> dict:
    """
    Bootstrap no paramétrico: se remuestrean CLIENTES (las filas de los
    arreglos, todos a la vez) y se recalcula el estadístico. Intervalo percentil.
    """
    arreglos = [np.asarray(a) for a in arreglos]
    n = len(arreglos[0])
    rng = np.random.default_rng(semilla)
    valores = np.empty(repeticiones)
    for b in range(repeticiones):
        idx = rng.integers(0, n, n)
        valores[b] = estadistico(*(a[idx] for a in arreglos))
    alfa = (1 - nivel) / 2
    return {
        "estimacion": float(estadistico(*arreglos)),
        "ic_inf": float(np.quantile(valores, alfa)),
        "ic_sup": float(np.quantile(valores, 1 - alfa)),
        "error_estandar": float(valores.std(ddof=1)),
    }


def diferencia_de_medias(y, t) -> float:
    y, t = np.asarray(y, dtype=float), np.asarray(t)
    return float(y[t == 1].mean() - y[t == 0].mean())


def efecto_bootstrap(y, t, repeticiones: int = 1000, nivel: float = 0.95, semilla: int = SEMILLA) -> dict:
    """
    Efecto promedio con intervalo bootstrap percentil. Sirve para comparar con
    el intervalo normal: con ~200 mil clientes tienen que coincidir casi
    exactamente (el TCL ya funciona); si no coincidieran, algo estaría mal.
    """
    r = bootstrap(diferencia_de_medias, y, t, repeticiones=repeticiones, nivel=nivel, semilla=semilla)
    return {"efecto": r["estimacion"], "error_estandar": r["error_estandar"],
            "ic_inf": r["ic_inf"], "ic_sup": r["ic_sup"]}


# ─── Modelos de uplift (etapa 3) ─────────────────────────────────────────────
def curva_qini(y, t, puntaje) -> pd.DataFrame:
    """
    Curva Qini: se ordena a los clientes por uplift predicho (mayor primero) y,
    para cada k, se estiman las compras incrementales de tratar a los k primeros
    comparando tratados y control DENTRO de esos k:

        incremental(k) = (Y_T(k) / N_T(k) - Y_C(k) / N_C(k)) · k / n,

    con Y_T(k), N_T(k) compras y cantidad de tratados entre los k primeros (y lo
    mismo para el control). Es la Q(k) de Radcliffe (2007) reescalada por
    k / N_T(k), para que quede en compras incrementales por cliente de la
    población y el último punto sea el efecto promedio. Con asignación 50/50 el
    orden entre modelos es el mismo que con la Q(k) original.
    Solo usa (y, t) observados: sirve también con datos reales.
    """
    y, t = np.asarray(y, dtype=float), np.asarray(t)
    orden = np.argsort(-np.asarray(puntaje, dtype=float), kind="stable")
    y, t = y[orden], t[orden]
    n_t, n_c = np.cumsum(t == 1), np.cumsum(t == 0)
    y_t, y_c = np.cumsum(y * (t == 1)), np.cumsum(y * (t == 0))
    n = len(y)
    k = np.arange(1, n + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dif = np.where((n_t > 0) & (n_c > 0), y_t / n_t - y_c / n_c, 0.0)
    return pd.DataFrame({"fraccion": np.r_[0.0, k / n], "incremental": np.r_[0.0, dif * k / n]})


def coeficiente_qini(y, t, puntaje) -> float:
    """
    Área entre la curva Qini y la recta de tratar al azar (que une el origen con
    el efecto de tratar a todos). 0 = el modelo no ordena mejor que el azar;
    negativo = ordena al revés. Está en las mismas unidades que la curva.
    """
    c = curva_qini(y, t, puntaje)
    azar = c["fraccion"] * c["incremental"].iloc[-1]
    d, x = (c["incremental"] - azar).to_numpy(), c["fraccion"].to_numpy()
    return float(np.sum((d[1:] + d[:-1]) / 2 * np.diff(x)))      # regla del trapecio


def pehe(tau_real, tau_estimado) -> float:
    """
    Raíz del error cuadrático medio del efecto individual (PEHE, Hill 2011).
    Solo se puede calcular con el efecto real, es decir, en simulación.
    """
    d = np.asarray(tau_real, dtype=float) - np.asarray(tau_estimado, dtype=float)
    return float(np.sqrt(np.mean(d ** 2)))


def captura_politica(tau_real, tau_estimado, fraccion: float = 0.3) -> float:
    """
    Si se trata al `fraccion` con mayor uplift ESTIMADO, qué parte de lo que
    ganaría la mejor política posible (tratar al `fraccion` con mayor uplift
    REAL) se obtiene. 1 = igual que el oráculo; ~fraccion·ATE / óptimo = azar.
    Es la métrica que importa para la decisión de negocio: no hace falta estimar
    bien el efecto de cada cliente, alcanza con ordenarlos bien.
    """
    tau_real = np.asarray(tau_real, dtype=float)
    k = max(1, int(round(fraccion * len(tau_real))))
    elegidos = np.argsort(-np.asarray(tau_estimado, dtype=float), kind="stable")[:k]
    optimo = np.sort(tau_real)[::-1][:k].sum()
    return float(tau_real[elegidos].sum() / optimo) if optimo > 0 else float("nan")