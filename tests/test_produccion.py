"""Tests de las etapas 6 y 7: validación, drift, control permanente y potencia."""

import numpy as np
import pandas as pd
import pytest

from src import carga, monitoreo, potencia
from src.pipeline import en_control_permanente


# ─── Validación y drift ──────────────────────────────────────────────────────
@pytest.fixture
def referencia():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"client_id": [f"c{i}" for i in range(5000)],
                         "a": rng.normal(size=5000), "b": rng.exponential(size=5000)})


def test_validacion_acepta_datos_iguales_y_frena_los_rotos(referencia):
    perfil = monitoreo.perfil(referencia, ["a", "b"])
    assert len(monitoreo.validar(referencia, perfil)) == 0

    sin_columna = referencia.drop(columns="b")
    assert "falta la columna" in monitoreo.validar(sin_columna, perfil)["mensaje"].tolist()

    nulos = referencia.copy()
    nulos.loc[:2000, "a"] = np.nan
    assert (monitoreo.validar(nulos, perfil)["nivel"] == "error").any()

    texto = referencia.assign(a=referencia["a"].astype(str))
    assert (monitoreo.validar(texto, perfil)["nivel"] == "error").any()

    duplicados = pd.concat([referencia, referencia.head(3)])
    assert (monitoreo.validar(duplicados, perfil)["variable"] == "client_id").any()

    escala = referencia.assign(b=referencia["b"] * 100)       # p. ej. cambio de unidades
    assert (monitoreo.validar(escala, perfil)["nivel"] == "aviso").any()


def test_psi_sin_drift_es_chico_y_con_drift_grande():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=20_000), rng.normal(size=20_000)
    assert monitoreo.psi(a, b) < 0.01
    assert monitoreo.psi(a, b + 0.5) > 0.2
    # faltantes nuevos cuentan como drift
    b_nulos = b.copy()
    b_nulos[:5000] = np.nan
    assert monitoreo.psi(a, b_nulos) > 0.25


def test_psi_esperado_sin_drift_coincide_con_la_simulacion():
    rng = np.random.default_rng(1)
    psis = [monitoreo.psi(rng.normal(size=500), rng.normal(size=500)) for _ in range(300)]
    assert np.mean(psis) == pytest.approx(monitoreo.psi_esperado_sin_drift(500, 500), rel=0.2)


def test_wasserstein_mide_el_corrimiento_en_desvios():
    rng = np.random.default_rng(2)
    a = rng.normal(10, 2, 50_000)
    assert monitoreo.wasserstein_estandarizado(a, a + 1) == pytest.approx(0.5, abs=0.02)


def test_sliced_wasserstein_ve_cambios_que_las_marginales_no():
    rng = np.random.default_rng(3)
    n = 20_000
    x = rng.normal(size=n)
    ref = pd.DataFrame({"x": x, "y": x + 0.1 * rng.normal(size=n)})       # muy correlacionadas
    x2 = rng.normal(size=n)
    act = pd.DataFrame({"x": x2, "y": -x2 + 0.1 * rng.normal(size=n)})    # mismas marginales, otra relación
    assert monitoreo.psi(ref["y"], act["y"]) < 0.02
    assert monitoreo.sliced_wasserstein(ref, act) > 5 * monitoreo.sliced_wasserstein(ref, ref.sample(frac=1, random_state=0))


def test_control_permanente_es_estable_y_no_depende_de_la_muestra_de_carga():
    ids = [str(i) for i in range(200_000)]
    control = np.array([en_control_permanente(i) for i in ids])
    assert control.mean() == pytest.approx(0.10, abs=0.005)
    assert all(en_control_permanente(i) == c for i, c in zip(ids[:1000], control[:1000]))
    muestra = np.array([carga.cliente_en_muestra(i, 0.2) for i in ids])
    # dentro de la muestra del 20%, el control sigue siendo ~10%
    assert control[muestra].mean() == pytest.approx(0.10, abs=0.01)


# ─── Potencia ────────────────────────────────────────────────────────────────
def test_tamano_y_mde_son_inversos():
    n = potencia.tamano_por_grupo(0.1, 1.0)
    assert n == 1570                                      # 2 (1.96 + 0.84)^2 / 0.01
    assert potencia.efecto_minimo_detectable(n, 1.0) == pytest.approx(0.1, rel=0.01)
    assert potencia.potencia_teorica(n, 0.1, 1.0) == pytest.approx(0.8, abs=0.01)


def test_cuped_achica_la_muestra_en_un_factor_uno_menos_rho_cuadrado():
    n = potencia.tamano_por_grupo(0.1, 1.0)
    n_c = potencia.tamano_por_grupo(0.1, potencia.desvio_cuped(1.0, 0.6))
    assert n_c / n == pytest.approx(1 - 0.36, rel=0.01)


def test_potencia_simulada_coincide_con_la_formula():
    rng = np.random.default_rng(0)
    a, b = rng.exponential(1.0, 100_000) + 0.1, rng.exponential(1.0, 100_000)   # asimétricas
    n = potencia.tamano_por_grupo(0.1, a.std(), b.std())
    assert potencia.potencia_simulada(a, b, n, repeticiones=400) == pytest.approx(0.8, abs=0.07)


def test_srm():
    assert potencia.srm([5000, 5040]) > 0.05
    assert potencia.srm([5000, 5600]) < 0.001


def test_espiar_infla_los_falsos_positivos():
    una = potencia.falsos_positivos_espiando(500, 1, repeticiones=2000)
    diez = potencia.falsos_positivos_espiando(500, 10, repeticiones=2000)
    assert una == pytest.approx(0.05, abs=0.015)
    assert diez > 0.12