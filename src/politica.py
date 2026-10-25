"""
Etapa 5: a quién darle el incentivo con un presupuesto B.

Economía de un cliente i (supuestos en Supuestos):

    monto esperado de una compra:  M_i  (su ticket medio)
    ganancia incremental:          v_i = (mu1_i - mu0_i) · margen · M_i
    gasto esperado del incentivo:  c_i = mu1_i · descuento · M_i + costo_contacto

El descuento se paga en TODAS las compras de los tratados, también en las que
iban a ocurrir igual (mu0_i): por eso tratar a quien compra siempre cuesta
mucho y no suma nada. El problema es una mochila:

    max  sum_i v_i x_i   sujeto a  sum_i c_i x_i <= B,   x_i en {0, 1}.

La relajación lineal (0 <= x_i <= 1) se resuelve ordenando por v_i / c_i y
llenando hasta agotar el presupuesto; la razón del primer cliente que no entra
es el precio sombra lambda: cuánta ganancia incremental aporta un peso más de
presupuesto. Si lambda < 1, el último peso gastado devuelve menos de un peso:
el presupuesto es más grande de lo que conviene.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from .config import SEMILLA


@dataclass(frozen=True)
class Supuestos:
    """X5 no publica márgenes ni costos: son supuestos, con análisis de sensibilidad."""
    margen: float = 0.25           # margen bruto sobre el monto de la compra
    descuento: float = 0.10        # parte del ticket que se devuelve si el tratado compra
    costo_contacto: float = 1.0    # costo fijo por cliente tratado (envío), en la moneda de los tickets


def monto_esperado(ticket_medio) -> np.ndarray:
    """Ticket medio del cliente; sin historial, la mediana de los que tienen."""
    m = np.asarray(ticket_medio, dtype=float)
    return np.where(np.isfinite(m), m, np.nanmedian(m))


def valor_y_costo(mu0, mu1, monto, sup: Supuestos) -> tuple[np.ndarray, np.ndarray]:
    mu0, mu1, monto = (np.asarray(a, dtype=float) for a in (mu0, mu1, monto))
    v = (mu1 - mu0) * sup.margen * monto
    c = mu1 * sup.descuento * monto + sup.costo_contacto
    return v, c


@dataclass
class Solucion:
    elegidos: np.ndarray          # bool, a quién tratar
    valor: float                  # sum v_i de los elegidos
    costo: float                  # sum c_i de los elegidos
    precio_sombra: float          # lambda; 0 si el presupuesto no limita


def _llenar(orden: np.ndarray, c: np.ndarray, presupuesto: float) -> tuple[np.ndarray, int | None]:
    """
    Recorre `orden` y trata a cada cliente mientras entre en el presupuesto; los
    que no entran se saltean (alguno posterior más barato puede entrar).
    Devuelve los elegidos y la posición en `orden` del primero que no entró.
    """
    elegidos = np.zeros(len(c), dtype=bool)
    acumulado, primero_afuera = 0.0, None
    for pos, i in enumerate(orden):
        if acumulado + c[i] <= presupuesto:
            elegidos[i] = True
            acumulado += c[i]
        elif primero_afuera is None:
            primero_afuera = pos
    return elegidos, primero_afuera


def mochila(v, c, presupuesto: float) -> Solucion:
    """Greedy por v_i / c_i entre los clientes con v_i > 0."""
    v, c = np.asarray(v, dtype=float), np.asarray(c, dtype=float)
    if (c <= 0).any():
        raise ValueError("Los costos tienen que ser positivos")
    candidatos = np.flatnonzero(v > 0)
    orden = candidatos[np.argsort(-(v[candidatos] / c[candidatos]), kind="stable")]
    elegidos, primero_afuera = _llenar(orden, c, presupuesto)
    lam = float(v[orden[primero_afuera]] / c[orden[primero_afuera]]) if primero_afuera is not None else 0.0
    return Solucion(elegidos, float(v[elegidos].sum()), float(c[elegidos].sum()), lam)


def mochila_exacta(v, c, presupuesto: float) -> Solucion:
    """Solución entera exacta con programación lineal entera (para comparar con el greedy)."""
    v, c = np.asarray(v, dtype=float), np.asarray(c, dtype=float)
    n = len(v)
    r = milp(-v, constraints=LinearConstraint(c[None, :], -np.inf, presupuesto),
             integrality=np.ones(n), bounds=Bounds(0, 1))
    if not r.success:
        raise RuntimeError(f"milp no convergió: {r.message}")
    elegidos = r.x > 0.5
    return Solucion(elegidos, float(v[elegidos].sum()), float(c[elegidos].sum()), float("nan"))


def por_puntaje(puntaje, c, presupuesto: float) -> np.ndarray:
    """Trata de mayor a menor puntaje hasta agotar el presupuesto (ignora el valor)."""
    orden = np.argsort(-np.asarray(puntaje, dtype=float), kind="stable")
    return _llenar(orden, np.asarray(c, dtype=float), presupuesto)[0]


def al_azar(c, presupuesto: float, semilla: int = SEMILLA) -> np.ndarray:
    orden = np.random.default_rng(semilla).permutation(len(c))
    return _llenar(orden, np.asarray(c, dtype=float), presupuesto)[0]


def camino(v, c) -> pd.DataFrame:
    """
    Soluciones de la relajación lineal para todos los presupuestos a la vez.
    """
    v, c = np.asarray(v, dtype=float), np.asarray(c, dtype=float)
    candidatos = np.flatnonzero(v > 0)
    orden = candidatos[np.argsort(-(v[candidatos] / c[candidatos]), kind="stable")]
    return pd.DataFrame({
        "cliente": orden,
        "razon": v[orden] / c[orden],
        "costo_acumulado": np.cumsum(c[orden]),
        "valor_acumulado": np.cumsum(v[orden]),
    })


def presupuesto_optimo(v, c) -> float:
    """
    Presupuesto que maximiza la ganancia neta esperada sum (v_i - c_i): se trata
    mientras la razón v_i / c_i sea mayor que 1, es decir, hasta que el precio
    sombra cae a 1.
    """
    cam = camino(v, c)
    rentables = cam[cam["razon"] > 1]
    return float(rentables["costo_acumulado"].iloc[-1]) if len(rentables) else 0.0