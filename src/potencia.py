"""
Etapa 7: potencia y controles del A/B que validaría la política.

Dos grupos al azar (A: política nueva, B: la de comparación), métrica por
cliente con desvíos sigma_A y sigma_B, test bilateral de nivel alfa:

    error estándar de la diferencia:  se = sqrt((sigma_A^2 + sigma_B^2) / n)
    efecto mínimo detectable:         MDE = (z_{1-alfa/2} + z_{potencia}) · se
    clientes por grupo:               n = (z_{1-alfa/2} + z_{potencia})^2 (sigma_A^2 + sigma_B^2) / delta^2

Con CUPED (una covariable previa con correlación rho con la métrica), la
varianza se multiplica por 1 - rho^2, y el tamaño de muestra también.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from .config import SEMILLA


def _z(alfa: float, potencia: float) -> float:
    return float(stats.norm.ppf(1 - alfa / 2) + stats.norm.ppf(potencia))


def tamano_por_grupo(delta: float, desvio_a: float, desvio_b: float | None = None,
                     alfa: float = 0.05, potencia: float = 0.8) -> float:
    desvio_b = desvio_a if desvio_b is None else desvio_b
    if delta == 0:
        return math.inf
    return math.ceil(_z(alfa, potencia) ** 2 * (desvio_a ** 2 + desvio_b ** 2) / delta ** 2)


def efecto_minimo_detectable(n: int, desvio_a: float, desvio_b: float | None = None,
                             alfa: float = 0.05, potencia: float = 0.8) -> float:
    desvio_b = desvio_a if desvio_b is None else desvio_b
    return _z(alfa, potencia) * math.sqrt((desvio_a ** 2 + desvio_b ** 2) / n)


def potencia_teorica(n, delta: float, desvio_a: float, desvio_b: float | None = None,
                     alfa: float = 0.05) -> np.ndarray:
    desvio_b = desvio_a if desvio_b is None else desvio_b
    se = np.sqrt((desvio_a ** 2 + desvio_b ** 2) / np.asarray(n, dtype=float))
    z = stats.norm.ppf(1 - alfa / 2)
    return stats.norm.cdf(abs(delta) / se - z) + stats.norm.cdf(-abs(delta) / se - z)


def desvio_cuped(desvio: float, rho: float) -> float:
    return desvio * math.sqrt(1 - rho ** 2)


def srm(conteos, proporciones=None) -> float:
    """
    Sample ratio mismatch: p-valor chi-cuadrado de que los grupos tengan las
    proporciones planeadas. Un p-valor muy chico (p. ej. < 0.001) indica que la
    asignación o el registro fallaron, y el resultado del A/B no es confiable.
    """
    conteos = np.asarray(conteos, dtype=float)
    proporciones = np.full(len(conteos), 1 / len(conteos)) if proporciones is None else np.asarray(proporciones)
    return float(stats.chisquare(conteos, conteos.sum() * proporciones).pvalue)


def potencia_simulada(muestra_a, muestra_b, n: int, repeticiones: int = 500, alfa: float = 0.05,
                      semilla: int = SEMILLA) -> float:
    """
    Potencia por simulación: se remuestrean n clientes por grupo de las
    distribuciones observadas y se aplica un test t de Welch. Verifica la
    fórmula normal con la distribución real de la métrica (asimétrica, con
    muchos ceros), sin suponer normalidad.
    """
    a, b = np.asarray(muestra_a, dtype=float), np.asarray(muestra_b, dtype=float)
    rng = np.random.default_rng(semilla)
    rechazos = 0
    for _ in range(repeticiones):
        xa, xb = rng.choice(a, n), rng.choice(b, n)
        rechazos += stats.ttest_ind(xa, xb, equal_var=False).pvalue < alfa
    return rechazos / repeticiones


def falsos_positivos_espiando(n_por_grupo: int, espiadas: int, repeticiones: int = 2000,
                              alfa: float = 0.05, semilla: int = SEMILLA) -> float:
    """
    A/A (sin efecto) mirando el resultado `espiadas` veces durante el
    experimento y frenando en la primera p < alfa. Con una sola mirada, la tasa
    de falsos positivos es alfa; con más, crece.
    """
    rng = np.random.default_rng(semilla)
    cortes = np.linspace(n_por_grupo / espiadas, n_por_grupo, espiadas).astype(int)
    z = stats.norm.ppf(1 - alfa / 2)
    positivos = 0
    for _ in range(repeticiones):
        a, b = rng.normal(size=n_por_grupo), rng.normal(size=n_por_grupo)
        ca, cb = np.cumsum(a), np.cumsum(b)
        dif = ca[cortes - 1] / cortes - cb[cortes - 1] / cortes
        positivos += bool(np.any(np.abs(dif) / np.sqrt(2 / cortes) > z))
    return positivos / repeticiones