"""
Datos semisintéticos con efecto conocido (etapa 3).

En los datos reales nunca se observa el efecto individual, así que no se puede
medir directamente qué método lo estima mejor. Acá se usan las variables reales
de los clientes, pero los resultados se generan con funciones elegidas:

    logit mu0(x) = b + 0.8 frec - 0.6 rec + 0.2 ticket + 0.3 canje
    logit mu1(x) = logit mu0(x) + efecto_logit(x)
    tau(x) = mu1(x) - mu0(x)

El efecto en escala logit es a · exp(heterogeneidad · h(x) / 2): positivo para
todos, mayor en clientes de frecuencia media y en los que usan el programa de
puntos, y constante si heterogeneidad = 0. Además hay un grupo "no molestar"
(clientes inactivos que no canjean) con efecto negativo.
Las constantes se calibran para que la tasa de compra sin incentivo y el efecto
promedio sean los pedidos en cada escenario.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit

from .config import SEMILLA

COLUMNAS_GENERADORAS = ["n_transacciones", "dias_desde_ultima", "ticket_medio", "canjeo_alguna_vez"]

EFECTO_NO_MOLESTAR_LOGIT = -0.5


@dataclass(frozen=True)
class Escenario:
    nombre: str
    n: int = 20_000                    
    efecto_promedio: float = 0.03      
    heterogeneidad: float = 1.0        
    frac_no_molestar: float = 0.10     
    tasa_base: float = 0.60            


ESCENARIOS = [
    Escenario("base"),
    Escenario("efecto chico", efecto_promedio=0.01),
    Escenario("sin heterogeneidad", heterogeneidad=0.0, frac_no_molestar=0.0),
    Escenario("mucha heterogeneidad", heterogeneidad=2.5),
    Escenario("sin no molestar", frac_no_molestar=0.0),
    Escenario("muestra chica", n=5_000),
    Escenario("muestra grande", n=80_000),
]


@dataclass
class DatosSimulados:
    X: pd.DataFrame
    t: np.ndarray
    y: np.ndarray
    tau: np.ndarray                    
    mu0: np.ndarray
    mu1: np.ndarray
    no_molestar: np.ndarray = field(repr=False)


def _estandarizar(x: np.ndarray) -> np.ndarray:
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / s if s > 0 else np.zeros_like(x)


def componentes(X: pd.DataFrame) -> pd.DataFrame:
    """Las variables del proceso generador, transformadas y estandarizadas."""
    faltan = set(COLUMNAS_GENERADORAS) - set(X.columns)
    if faltan:
        raise ValueError(f"Faltan columnas para simular: {sorted(faltan)}")
    rec = np.log1p(X["dias_desde_ultima"].to_numpy(dtype=float))
    rec = np.where(np.isnan(rec), np.nanmax(rec) if np.isfinite(np.nanmax(rec)) else 0.0, rec)  # sin compras = muy inactivo
    ticket = _estandarizar(np.log1p(X["ticket_medio"].to_numpy(dtype=float)))
    return pd.DataFrame(
        {
            "frec": _estandarizar(np.log1p(X["n_transacciones"].to_numpy(dtype=float))),
            "rec": _estandarizar(rec),
            "ticket": np.nan_to_num(ticket),
            "canje": np.nan_to_num(X["canjeo_alguna_vez"].to_numpy(dtype=float)),
        },
        index=X.index,
    )


def _calibrar(funcion, objetivo: float, nombre: str) -> float:
    """Busca la constante c tal que funcion(c) = objetivo."""
    try:
        return brentq(lambda c: funcion(c) - objetivo, -10, 10)
    except ValueError as e:
        raise ValueError(f"No se puede alcanzar {nombre} = {objetivo} con este escenario") from e


def simular(X_real: pd.DataFrame, escenario: Escenario, semilla: int = SEMILLA) -> DatosSimulados:
    rng = np.random.default_rng(semilla)
    reemplazo = escenario.n > len(X_real)
    idx = rng.choice(len(X_real), size=escenario.n, replace=reemplazo)
    X = X_real.iloc[idx].reset_index(drop=True)
    z = componentes(X)

    lineal = 0.8 * z["frec"] - 0.6 * z["rec"] + 0.2 * z["ticket"] + 0.3 * z["canje"]
    b = _calibrar(lambda c: expit(c + lineal).mean(), escenario.tasa_base, "la tasa base")
    logit0 = (b + lineal).to_numpy()

    h = np.exp(-0.5 * z["frec"] ** 2) + 0.5 * z["canje"]
    h = _estandarizar(h.to_numpy())

    no_molestar = np.zeros(escenario.n, dtype=bool)
    if escenario.frac_no_molestar > 0:
        puntaje = z["rec"].to_numpy() - 2.0 * z["canje"].to_numpy()
        corte = np.quantile(puntaje, 1 - escenario.frac_no_molestar)
        no_molestar = puntaje >= corte

    def efecto_logit(a: float) -> np.ndarray:
        e = a * np.exp(0.5 * escenario.heterogeneidad * h)
        return np.where(no_molestar, EFECTO_NO_MOLESTAR_LOGIT, e)

    a = _calibrar(
        lambda c: (expit(logit0 + efecto_logit(c)) - expit(logit0)).mean(),
        escenario.efecto_promedio, "el efecto promedio",
    )
    mu0 = expit(logit0)
    mu1 = expit(logit0 + efecto_logit(a))

    t = rng.integers(0, 2, escenario.n)
    y = (rng.random(escenario.n) < np.where(t == 1, mu1, mu0)).astype(int)
    return DatosSimulados(X=X, t=t, y=y, tau=mu1 - mu0, mu0=mu0, mu1=mu1, no_molestar=no_molestar)