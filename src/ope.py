
"""
Etapa 5: evaluación off-policy con los datos del experimento.

Una política pi(x) en {0, 1} dice a quién tratar. Su valor es la ganancia
media por cliente si se aplicara a toda la población. En el experimento cada
cliente recibió un tratamiento al azar con probabilidad conocida, así que se
puede estimar ese valor sin aplicar la política:

    IPW:  V(pi) = (1/n) sum_i 1{T_i = pi(x_i)} / P(T = T_i) · R_i

Solo cuentan los clientes cuyo tratamiento coincide con el que habría elegido
la política, reponderados por la probabilidad de haberlo recibido. Es insesgado
porque la asignación fue al azar, pero tira la mitad de los datos.

    DR:   V(pi) = (1/n) sum_i [ q(x_i, pi_i) + 1{T_i = pi_i} / P(T = T_i) · (R_i - q(x_i, T_i)) ]

con q(x, a) la ganancia que predice el modelo si se aplica a. Si el modelo es
bueno, el término de corrección es chico y la varianza baja; si es malo, la
corrección lo arregla en promedio (doble robustez: alcanza con que P(T) sea la
correcta, y acá lo es por diseño).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SEMILLA
from .politica import Supuestos


def recompensa(y, t, monto, sup: Supuestos) -> np.ndarray:
    """Ganancia neta observada: margen de la compra, menos descuento y contacto si fue tratado."""
    y, t, monto = (np.asarray(a, dtype=float) for a in (y, t, monto))
    return y * monto * (sup.margen - sup.descuento * t) - sup.costo_contacto * t


def gasto(y, t, monto, sup: Supuestos) -> np.ndarray:
    """Gasto observado del incentivo (lo que sale del presupuesto)."""
    y, t, monto = (np.asarray(a, dtype=float) for a in (y, t, monto))
    return t * (y * sup.descuento * monto + sup.costo_contacto)


def modelo_recompensa(mu0, mu1, monto, sup: Supuestos) -> tuple[np.ndarray, np.ndarray]:
    """q(x, 0) y q(x, 1): ganancia esperada sin y con incentivo."""
    mu0, mu1, monto = (np.asarray(a, dtype=float) for a in (mu0, mu1, monto))
    return mu0 * monto * sup.margen, mu1 * monto * (sup.margen - sup.descuento) - sup.costo_contacto


def modelo_gasto(mu1, monto, sup: Supuestos) -> tuple[np.ndarray, np.ndarray]:
    mu1, monto = np.asarray(mu1, dtype=float), np.asarray(monto, dtype=float)
    return np.zeros_like(mu1), mu1 * sup.descuento * monto + sup.costo_contacto


def contribuciones(r, t, pi, p: float, q0=None, q1=None) -> np.ndarray:
    """
    Aporte de cada cliente al estimador; su media es el valor de la política.
    Sin q0/q1, IPW; con q0/q1, doblemente robusto.
    """
    r, t, pi = np.asarray(r, dtype=float), np.asarray(t), np.asarray(pi).astype(int)
    prob = np.where(t == 1, p, 1 - p)
    coincide = (t == pi) / prob
    if q0 is None:
        return coincide * r
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    q_pi = np.where(pi == 1, q1, q0)
    q_t = np.where(t == 1, q1, q0)
    return q_pi + coincide * (r - q_t)


def valor_ipw(r, t, pi, p: float) -> float:
    return float(contribuciones(r, t, pi, p).mean())


def valor_dr(r, t, pi, p: float, q0, q1) -> float:
    return float(contribuciones(r, t, pi, p, q0, q1).mean())


def evaluar(r, t, politicas: dict[str, np.ndarray], p: float, q0, q1, referencia: str,
            repeticiones: int = 1000, nivel: float = 0.95, semilla: int = SEMILLA) -> pd.DataFrame:
    """
    Valor IPW y DR de cada política, con IC bootstrap del DR 
    """
    nombres = list(politicas)
    ipw = np.column_stack([contribuciones(r, t, politicas[k], p) for k in nombres])
    dr = np.column_stack([contribuciones(r, t, politicas[k], p, q0, q1) for k in nombres])
    rng = np.random.default_rng(semilla)
    n = len(r)
    medias = np.empty((repeticiones, len(nombres)))
    for b in range(repeticiones):
        medias[b] = dr[rng.integers(0, n, n)].mean(axis=0)
    j = nombres.index(referencia)
    dif = medias - medias[:, [j]]
    alfa = (1 - nivel) / 2
    return pd.DataFrame({
        "politica": nombres,
        "tratados": [float(np.mean(politicas[k])) for k in nombres],
        "valor_ipw": ipw.mean(axis=0),
        "valor_dr": dr.mean(axis=0),
        "ic_inf": np.quantile(medias, alfa, axis=0),
        "ic_sup": np.quantile(medias, 1 - alfa, axis=0),
        "dif_vs_referencia": (dr - dr[:, [j]]).mean(axis=0),
        "dif_ic_inf": np.quantile(dif, alfa, axis=0),
        "dif_ic_sup": np.quantile(dif, 1 - alfa, axis=0),
        "error_estandar_ipw": ipw.std(axis=0, ddof=1) / np.sqrt(n),
        "error_estandar_dr": dr.std(axis=0, ddof=1) / np.sqrt(n),
    })