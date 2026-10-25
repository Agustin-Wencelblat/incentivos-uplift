"""Tests de la etapa 3: simulador, modelos de uplift y métricas."""

import numpy as np
import pandas as pd
import pytest

from src import evaluacion, uplift
from src.config import VARIABLES_MODELO
from src.simulacion import Escenario, simular


def clientes_falsos(n=20_000, semilla=0) -> pd.DataFrame:
    """Variables con la forma de las reales (sin base de datos)."""
    rng = np.random.default_rng(semilla)
    X = pd.DataFrame(rng.normal(size=(n, len(VARIABLES_MODELO))), columns=VARIABLES_MODELO)
    X["n_transacciones"] = rng.poisson(8, n).astype(float)
    X["dias_desde_ultima"] = rng.exponential(20, n)
    X["ticket_medio"] = rng.lognormal(6, 0.5, n)
    X["canjeo_alguna_vez"] = (rng.random(n) < 0.4).astype(float)
    sin_compras = X["n_transacciones"] == 0
    X.loc[sin_compras, ["dias_desde_ultima", "ticket_medio"]] = np.nan
    return X


@pytest.fixture(scope="module")
def X_real():
    return clientes_falsos()


# ─── Simulador ───────────────────────────────────────────────────────────────
def test_simulacion_respeta_lo_pedido(X_real):
    esc = Escenario("test", n=30_000, efecto_promedio=0.04, tasa_base=0.5, frac_no_molestar=0.1)
    d = simular(X_real, esc, semilla=1)
    assert d.tau.mean() == pytest.approx(0.04, abs=1e-6)
    assert d.mu0.mean() == pytest.approx(0.5, abs=1e-6)
    assert d.t.mean() == pytest.approx(0.5, abs=0.02)
    assert d.no_molestar.mean() == pytest.approx(0.1, abs=0.01)
    assert (d.tau[d.no_molestar] < 0).all()
    assert (d.tau[~d.no_molestar] > 0).all()
    # el efecto estimado con la diferencia de medias tiene que estar cerca del real
    assert evaluacion.diferencia_de_medias(d.y, d.t) == pytest.approx(0.04, abs=0.025)


def test_sin_heterogeneidad_el_efecto_casi_no_varia(X_real):
    con = simular(X_real, Escenario("a", heterogeneidad=1.0, frac_no_molestar=0), semilla=2)
    sin = simular(X_real, Escenario("b", heterogeneidad=0.0, frac_no_molestar=0), semilla=2)
    # sin heterogeneidad en logit, solo varía por la no linealidad de la logística
    assert sin.tau.std() < 0.5 * con.tau.std()


def test_simulacion_es_reproducible(X_real):
    a = simular(X_real, Escenario("a", n=5_000), semilla=3)
    b = simular(X_real, Escenario("a", n=5_000), semilla=3)
    assert np.array_equal(a.y, b.y) and np.array_equal(a.tau, b.tau)


# ─── Métricas ────────────────────────────────────────────────────────────────
def test_variable_transformada_estima_el_efecto():
    rng = np.random.default_rng(0)
    n = 200_000
    t = (rng.random(n) < 0.3).astype(int)               # p distinta de 1/2 a propósito
    y = (rng.random(n) < np.where(t == 1, 0.25, 0.20)).astype(int)
    assert uplift.variable_transformada(y, t, 0.3).mean() == pytest.approx(0.05, abs=0.01)


def test_pehe_y_captura_del_oraculo():
    tau = np.linspace(-0.1, 0.2, 1000)
    assert evaluacion.pehe(tau, tau) == 0
    assert evaluacion.captura_politica(tau, tau, 0.3) == pytest.approx(1.0)
    assert evaluacion.captura_politica(tau, -tau, 0.3) < 0


def test_qini_ordenar_bien_es_positivo_y_al_reves_negativo():
    rng = np.random.default_rng(0)
    n = 40_000
    x = rng.random(n)
    t = rng.integers(0, 2, n)
    y = (rng.random(n) < 0.3 + 0.2 * x * t).astype(int)   # efecto creciente en x
    bien = evaluacion.coeficiente_qini(y, t, x)
    mal = evaluacion.coeficiente_qini(y, t, -x)
    azar = evaluacion.coeficiente_qini(y, t, rng.random(n))
    # efecto 0.2 x con x uniforme: área teórica 0.1 (1 - 1/3 - 1/2) ≈ 0.0167
    assert bien == pytest.approx(0.0167, abs=0.005)
    assert mal == pytest.approx(-0.0167, abs=0.005)
    assert abs(azar) < 0.005
    c = evaluacion.curva_qini(y, t, x)
    assert c["fraccion"].iloc[0] == 0 and c["fraccion"].iloc[-1] == 1
    # el último punto es el efecto de tratar a todos (por cliente de la población)
    assert c["incremental"].iloc[-1] == pytest.approx(evaluacion.diferencia_de_medias(y, t), abs=1e-12)


# ─── Modelos ─────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def datos_heterogeneos(X_real):
    esc = Escenario("test", n=20_000, efecto_promedio=0.05, heterogeneidad=2.5)
    d = simular(X_real, esc, semilla=4)
    prueba = np.random.default_rng(4).random(len(d.y)) < 0.3
    return d, prueba


# La variable transformada es mucho más ruidosa (Z vale ±1/p o 0): se le pide menos
CORRELACION_MINIMA = {"z": 0.05}


@pytest.mark.parametrize("clave", list(uplift.MODELOS))
def test_modelos_ordenan_mejor_que_el_azar(datos_heterogeneos, clave):
    d, prueba = datos_heterogeneos
    m = uplift.construir(clave).fit(d.X[~prueba], d.t[~prueba], d.y[~prueba])
    tau_hat = m.efecto(d.X[prueba])
    assert tau_hat.shape == (prueba.sum(),)
    assert np.isfinite(tau_hat).all()
    # correlación positiva con el efecto real y captura por encima del azar
    assert np.corrcoef(tau_hat, d.tau[prueba])[0, 1] > CORRELACION_MINIMA.get(clave, 0.3)
    azar = evaluacion.captura_politica(d.tau[prueba], np.random.default_rng(0).random(prueba.sum()), 0.3)
    assert evaluacion.captura_politica(d.tau[prueba], tau_hat, 0.3) > azar + 0.05


def test_t_learner_da_probabilidades(datos_heterogeneos):
    d, _ = datos_heterogeneos
    mu0, mu1 = uplift.TLearner().fit(d.X, d.t, d.y).respuestas(d.X.head(100))
    assert ((0 <= mu0) & (mu0 <= 1) & (0 <= mu1) & (mu1 <= 1)).all()


def test_modelo_desconocido():
    with pytest.raises(ValueError):
        uplift.construir("nada")