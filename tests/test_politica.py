"""Tests de las etapas 4 y 5: calibración, Qini bootstrap, mochila y evaluación off-policy."""

import numpy as np
import pytest

from src import evaluacion, ope, politica
from src.politica import Supuestos


# ─── Etapa 4 ─────────────────────────────────────────────────────────────────
def datos_con_efecto_lineal(n=40_000, semilla=0):
    rng = np.random.default_rng(semilla)
    x = rng.random(n)
    t = rng.integers(0, 2, n)
    y = (rng.random(n) < 0.3 + 0.2 * x * t).astype(int)
    return x, t, y


def test_calibracion_de_un_modelo_perfecto_cae_sobre_la_diagonal():
    x, t, y = datos_con_efecto_lineal()
    cal = evaluacion.calibracion_por_grupo(y, t, 0.2 * x, grupos=5)
    assert list(cal["grupo"]) == [1, 2, 3, 4, 5]
    assert cal["uplift_predicho"].is_monotonic_decreasing
    pendiente = np.polyfit(cal["uplift_predicho"], cal["efecto_observado"], 1)[0]
    assert pendiente == pytest.approx(1, abs=0.25)
    # un modelo que exagera el efecto al doble da pendiente ~ 1/2
    pendiente_doble = np.polyfit(evaluacion.calibracion_por_grupo(y, t, 0.4 * x, 5)["uplift_predicho"],
                                 cal["efecto_observado"], 1)[0]
    assert pendiente_doble == pytest.approx(0.5, abs=0.15)


def test_qini_bootstrap_pareado_distingue_mejor_que_por_separado():
    x, t, y = datos_con_efecto_lineal(n=10_000)
    ruido = x + np.random.default_rng(1).normal(0, 0.5, len(x))
    boot = evaluacion.qini_bootstrap(y, t, {"bueno": x, "ruidoso": ruido}, repeticiones=200)
    dif = boot["bueno"] - boot["ruidoso"]
    assert dif.std() < boot["bueno"].std()
    assert (dif > 0).mean() > 0.9


# ─── Mochila ─────────────────────────────────────────────────────────────────
def test_mochila_respeta_el_presupuesto_y_casi_iguala_al_optimo_exacto():
    rng = np.random.default_rng(0)
    v = rng.normal(1, 1, 300)
    c = rng.uniform(0.5, 2, 300)
    B = 50.0
    greedy = politica.mochila(v, c, B)
    exacta = politica.mochila_exacta(v, c, B)
    assert greedy.costo <= B + 1e-9 and exacta.costo <= B + 1e-9
    assert not greedy.elegidos[v <= 0].any()
    assert greedy.valor <= exacta.valor + 1e-9
    assert greedy.valor >= 0.97 * exacta.valor
    assert greedy.precio_sombra > 0


def test_precio_sombra_es_el_valor_marginal_del_presupuesto():
    rng = np.random.default_rng(1)
    v = rng.uniform(0, 1, 5000)
    c = rng.uniform(0.5, 1.5, 5000)
    B, dB = 500.0, 5.0
    a, b = politica.mochila(v, c, B), politica.mochila(v, c, B + dB)
    assert (b.valor - a.valor) / dB == pytest.approx(a.precio_sombra, rel=0.1)


def test_sin_restriccion_el_precio_sombra_es_cero():
    v, c = np.array([1.0, 2.0, -1.0]), np.array([1.0, 1.0, 1.0])
    sol = politica.mochila(v, c, 100)
    assert sol.precio_sombra == 0
    assert list(sol.elegidos) == [True, True, False]


def test_presupuesto_optimo_trata_solo_si_cada_peso_devuelve_mas_de_un_peso():
    v = np.array([3.0, 2.0, 1.5, 0.5])
    c = np.array([1.0, 1.0, 1.0, 1.0])
    assert politica.presupuesto_optimo(v, c) == 3.0      # el cuarto devuelve 0.5 por peso


def test_el_descuento_sobre_compras_que_iban_a_ocurrir_hace_caro_tratar_a_los_fieles():
    sup = Supuestos(margen=0.25, descuento=0.10, costo_contacto=0)
    v_fiel, c_fiel = politica.valor_y_costo(0.95, 0.96, 100, sup)
    v_pers, c_pers = politica.valor_y_costo(0.20, 0.50, 100, sup)
    assert v_fiel / c_fiel < 1 < v_pers / c_pers


# ─── Evaluación off-policy ───────────────────────────────────────────────────
def mundo(n=50_000, semilla=0):
    """Experimento simulado con efecto heterogéneo y el valor real de una política."""
    rng = np.random.default_rng(semilla)
    x = rng.random(n)
    mu0, mu1 = 0.3 + 0.2 * x, 0.3 + 0.2 * x + 0.3 * (x - 0.4)
    t = rng.integers(0, 2, n)
    y = (rng.random(n) < np.where(t == 1, mu1, mu0)).astype(int)
    monto = np.full(n, 100.0)
    sup = Supuestos(0.25, 0.10, 1.0)
    pi = x > 0.6
    q0, q1 = ope.modelo_recompensa(mu0, mu1, monto, sup)
    valor_real = float(np.where(pi, q1, q0).mean())
    return x, t, y, monto, sup, pi, mu0, mu1, valor_real


def test_ipw_y_dr_son_insesgados_y_dr_tiene_menos_varianza():
    est_ipw, est_dr = [], []
    for s in range(30):
        x, t, y, monto, sup, pi, mu0, mu1, real = mundo(n=10_000, semilla=s)
        r = ope.recompensa(y, t, monto, sup)
        # modelo de recompensa deliberadamente imperfecto
        q0, q1 = ope.modelo_recompensa(mu0 * 0.9, mu1 * 1.1, monto, sup)
        est_ipw.append(ope.valor_ipw(r, t, pi, 0.5) - real)
        est_dr.append(ope.valor_dr(r, t, pi, 0.5, q0, q1) - real)
    assert abs(np.mean(est_ipw)) < 3 * np.std(est_ipw) / np.sqrt(30)
    assert abs(np.mean(est_dr)) < 3 * np.std(est_dr) / np.sqrt(30)
    assert np.std(est_dr) < np.std(est_ipw)


def test_evaluar_da_diferencias_pareadas_contra_la_referencia():
    x, t, y, monto, sup, pi, mu0, mu1, _ = mundo()
    r = ope.recompensa(y, t, monto, sup)
    q0, q1 = ope.modelo_recompensa(mu0, mu1, monto, sup)
    tabla = ope.evaluar(r, t, {"pi": pi, "nadie": np.zeros(len(t), bool)}, 0.5, q0, q1, "nadie", repeticiones=200)
    fila = tabla.set_index("politica").loc["nadie"]
    assert fila["dif_vs_referencia"] == 0 and fila["dif_ic_inf"] == 0
    p = tabla.set_index("politica").loc["pi"]
    assert p["dif_ic_inf"] < p["dif_vs_referencia"] < p["dif_ic_sup"]
    assert p["dif_ic_sup"] - p["dif_ic_inf"] < p["ic_sup"] - p["ic_inf"]