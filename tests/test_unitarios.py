"""Tests de la etapa 2 (evaluación del experimento) y del muestreo de clientes."""

import numpy as np
import pandas as pd
import pytest

from src import carga, evaluacion


def experimento_simulado(n=20_000, efecto=0.05, semilla=0):
    """Experimento con asignación al azar, una covariable previa y efecto conocido."""
    rng = np.random.default_rng(semilla)
    x_pre = rng.poisson(10, n).astype(float)                       # por ejemplo, compras previas
    t = rng.integers(0, 2, n)
    p = np.clip(0.2 + 0.03 * x_pre + efecto * t, 0, 1)
    y = (rng.random(n) < p).astype(int)
    return x_pre, t, y


# ─── Efecto promedio ─────────────────────────────────────────────────────────
def test_efecto_promedio_es_la_diferencia_de_medias():
    y = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    t = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    r = evaluacion.efecto_promedio(y, t)
    assert r["efecto"] == pytest.approx(0.75 - 0.25)
    assert r["ic_inf"] < r["efecto"] < r["ic_sup"]


def test_intervalo_cubre_el_efecto_real_la_mayoria_de_las_veces():
    cubre = [
        (r := evaluacion.efecto_promedio(y, t))["ic_inf"] <= 0.05 <= r["ic_sup"]
        for _, t, y in (experimento_simulado(n=4000, semilla=s) for s in range(200))
    ]
    assert 0.90 < np.mean(cubre) < 0.99          # nivel nominal 95%


def test_bootstrap_coincide_con_el_intervalo_normal_en_muestras_grandes():
    _, t, y = experimento_simulado()
    normal = evaluacion.efecto_promedio(y, t)
    boot = evaluacion.efecto_bootstrap(y, t, repeticiones=400)
    assert boot["ic_inf"] == pytest.approx(normal["ic_inf"], abs=0.004)
    assert boot["ic_sup"] == pytest.approx(normal["ic_sup"], abs=0.004)


# ─── CUPED ───────────────────────────────────────────────────────────────────
def test_cuped_reduce_la_varianza_sin_sesgar():
    efectos_sin, efectos_con, ee_sin, ee_con = [], [], [], []
    for s in range(60):
        x_pre, t, y = experimento_simulado(n=5000, semilla=s)
        sin, con = evaluacion.efecto_promedio(y, t), evaluacion.efecto_cuped(y, t, x_pre)
        efectos_sin.append(sin["efecto"]); efectos_con.append(con["efecto"])
        ee_sin.append(sin["error_estandar"]); ee_con.append(con["error_estandar"])
    assert np.mean(efectos_con) == pytest.approx(0.05, abs=0.006)      # insesgado
    assert np.std(efectos_con) < np.std(efectos_sin)                    # menos varianza real
    assert np.mean(ee_con) < np.mean(ee_sin)                            # y menor error estándar


def test_cuped_con_covariable_posterior_al_tratamiento_sesga_el_efecto():
    # Si el tratamiento afecta a la covariable, el ajuste le "saca" parte del efecto.
    rng = np.random.default_rng(1)
    n = 50_000
    t = rng.integers(0, 2, n)
    x_post = rng.normal(0, 1, n) + 1.0 * t          # el tratamiento mueve la covariable
    y = (rng.random(n) < np.clip(0.4 + 0.1 * x_post, 0, 1)).astype(int)
    real = evaluacion.efecto_promedio(y, t)["efecto"]            # ~0.10, el efecto total
    ajustado = evaluacion.efecto_cuped(y, t, x_post)["efecto"]
    assert abs(ajustado - real) > 0.03


# ─── Balance y comparaciones múltiples ───────────────────────────────────────
def test_balance_detecta_asignacion_sesgada():
    rng = np.random.default_rng(0)
    n = 6000
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    al_azar = rng.integers(0, 2, n)
    sesgado = (rng.random(n) < 1 / (1 + np.exp(-1.5 * X["a"]))).astype(int)   # depende de a

    assert evaluacion.diferencia_estandarizada(X, al_azar)["smd"].abs().max() < 0.1
    assert evaluacion.auc_balance(X, al_azar) < 0.55
    smd = evaluacion.diferencia_estandarizada(X, sesgado).set_index("variable")["smd"]
    assert abs(smd["a"]) > 0.5 and abs(smd["b"]) < 0.1
    assert evaluacion.auc_balance(X, sesgado) > 0.7


def test_holm():
    ajustados = evaluacion.holm([0.01, 0.04, 0.03, 0.20])
    assert ajustados == pytest.approx([0.04, 0.09, 0.09, 0.20])
    assert np.all(ajustados >= [0.01, 0.04, 0.03, 0.20])


# ─── Carga ───────────────────────────────────────────────────────────────────
def test_muestra_de_clientes_es_deterministica_y_respeta_la_fraccion():
    ids = [f"{i:010x}" for i in range(20_000)]
    elegidos = [c for c in ids if carga.cliente_en_muestra(c, 0.2)]
    assert len(elegidos) / len(ids) == pytest.approx(0.2, abs=0.01)
    assert elegidos == [c for c in ids if carga.cliente_en_muestra(c, 0.2)]
