
"""
Modelos de uplift: estiman el efecto individual

    tau(x) = E[Y(1) - Y(0) | X = x]

Todos comparten la interfaz fit(X, t, y) -> self y efecto(X) -> array.
Los modelos base son HistGradientBoosting, que maneja faltantes sin imputar (los clientes sin compras tienen NaN en ticket, intervalos, etc.).

Supuesto común: tratamiento asignado al azar con probabilidad p constante, así que p se estima como la proporción 
de tratados y no hace falta un modelo de propensión. Con datos observacionales habría que modelar P(T = 1 | X).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from .config import SEMILLA

_PARAMETROS = dict(
    max_iter=500, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100,
    l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
)


def clasificador(semilla: int = SEMILLA) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=semilla, **_PARAMETROS)


def regresor(semilla: int = SEMILLA) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(random_state=semilla, **_PARAMETROS)


def _proba(modelo, X) -> np.ndarray:
    return modelo.predict_proba(X)[:, 1]


def variable_transformada(y, t, p: float) -> np.ndarray:
    """
    Z = Y (T / p - (1 - T) / (1 - p)).

    Con T independiente de (Y(0), Y(1)) dado X y P(T = 1) = p:
        E[Z | X] = p E[Y(1) | X] / p - (1 - p) E[Y(0) | X] / (1 - p) = tau(X).
    """
    y, t = np.asarray(y, dtype=float), np.asarray(t, dtype=float)
    return y * (t / p - (1 - t) / (1 - p))


# ─── Interfaz común ──────────────────────────────────────────────────────────
class ModeloUplift(ABC):
    nombre: str = ""

    def __init__(self, semilla: int = SEMILLA):
        self.semilla = semilla

    @abstractmethod
    def fit(self, X: pd.DataFrame, t, y) -> "ModeloUplift": ...

    @abstractmethod
    def efecto(self, X: pd.DataFrame) -> np.ndarray: ...


class SLearner(ModeloUplift):
    """
    Un solo modelo con el tratamiento como una variable más:
        tau(x) = f(x, 1) - f(x, 0).
    """

    nombre = "S-learner"

    def fit(self, X, t, y):
        self.modelo_ = clasificador(self.semilla).fit(X.assign(_t=np.asarray(t)), y)
        return self

    def efecto(self, X):
        return _proba(self.modelo_, X.assign(_t=1)) - _proba(self.modelo_, X.assign(_t=0))


class TLearner(ModeloUplift):
    """
    Un modelo con los tratados y otro con el control:
        tau(x) = mu1(x) - mu0(x).
    Cada modelo ve la mitad de los datos y sus errores no se cancelan.
    """

    nombre = "T-learner"

    def fit(self, X, t, y):
        t, y = np.asarray(t), np.asarray(y)
        self.mu0_ = clasificador(self.semilla).fit(X[t == 0], y[t == 0])
        self.mu1_ = clasificador(self.semilla).fit(X[t == 1], y[t == 1])
        return self

    def respuestas(self, X) -> tuple[np.ndarray, np.ndarray]:
        """(mu0(x), mu1(x)): probabilidad de compra sin y con incentivo."""
        return _proba(self.mu0_, X), _proba(self.mu1_, X)

    def efecto(self, X):
        mu0, mu1 = self.respuestas(X)
        return mu1 - mu0


class XLearner(ModeloUplift):
    nombre = "X-learner"

    def fit(self, X, t, y):
        t, y = np.asarray(t), np.asarray(y, dtype=float)
        base = TLearner(self.semilla).fit(X, t, y)
        mu0, mu1 = base.respuestas(X)
        d1 = y[t == 1] - mu0[t == 1]
        d0 = mu1[t == 0] - y[t == 0]
        self.tau1_ = regresor(self.semilla).fit(X[t == 1], d1)
        self.tau0_ = regresor(self.semilla).fit(X[t == 0], d0)
        self.p_ = t.mean()
        return self

    def efecto(self, X):
        return self.p_ * self.tau0_.predict(X) + (1 - self.p_) * self.tau1_.predict(X)


class VariableTransformada(ModeloUplift):
    nombre = "Variable transformada"

    def fit(self, X, t, y):
        t = np.asarray(t)
        self.modelo_ = regresor(self.semilla).fit(X, variable_transformada(y, t, t.mean()))
        return self

    def efecto(self, X):
        return self.modelo_.predict(X)


class DRLearner(ModeloUplift):
    nombre = "DR-learner"

    def __init__(self, semilla: int = SEMILLA, pliegues: int = 5):
        super().__init__(semilla)
        self.pliegues = pliegues

    def fit(self, X, t, y):
        t, y = np.asarray(t), np.asarray(y, dtype=float)
        p = t.mean()
        psi = np.empty(len(y))
        for entrena, evalua in KFold(self.pliegues, shuffle=True, random_state=self.semilla).split(X):
            base = TLearner(self.semilla).fit(X.iloc[entrena], t[entrena], y[entrena])
            mu0, mu1 = base.respuestas(X.iloc[evalua])
            te, ye = t[evalua], y[evalua]
            psi[evalua] = mu1 - mu0 + te * (ye - mu1) / p - (1 - te) * (ye - mu0) / (1 - p)
        self.modelo_ = regresor(self.semilla).fit(X, psi)
        return self

    def efecto(self, X):
        return self.modelo_.predict(X)


MODELOS = {
    "s": SLearner,
    "t": TLearner,
    "x": XLearner,
    "z": VariableTransformada,
    "dr": DRLearner,
}


def construir(clave: str, semilla: int = SEMILLA) -> ModeloUplift:
    if clave not in MODELOS:
        raise ValueError(f"Modelo desconocido: {clave}. Opciones: {', '.join(MODELOS)}")
    return MODELOS[clave](semilla)

# TODO (etapa 4): ajustar hiperparámetros con una métrica de uplift (Qini por validación cruzada).
# TODO (no es necesario, lo hago si me queda tiempo): causal forest con econml (CausalForestDML).