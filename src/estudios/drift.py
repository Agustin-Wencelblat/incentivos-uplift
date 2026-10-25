"""
Etapa 6 — ¿Detecta el monitoreo los cambios que importan?

    python -m src.estudios.drift --rapido
    python -m src.estudios.drift

X5 es una sola campaña, así que no hay "mes siguiente" con el que comparar. Se
parte la población en dos mitades al azar: una es la referencia (como en
entrenamiento) y a la otra se le aplican cambios conocidos. Se mira si la
validación frena lo que tiene que frenar, si el PSI y Wasserstein marcan las
variables correctas y si el cambio llega a las predicciones del modelo.

Escenarios:
    sin cambios         la otra mitad tal cual (no debería haber alertas)
    inflación 30%       montos y puntos multiplicados por 1.3
    clientes inactivos  30 días más desde la última compra y la mitad de gasto reciente
    más clientes nuevos 30% de la muestra reemplazado por clientes de baja frecuencia
    falla de datos      edad sin cargar en el 40% de los clientes

Además: con qué frecuencia el PSI supera los umbrales sin ningún cambio, según
el tamaño de muestra (falsas alarmas).

Genera en results/etapa6/: escenarios.csv, psi.png, falsas_alarmas.csv, resumen.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import datos, graficos, monitoreo, uplift
from ..config import SEMILLA, VARIABLES_MODELO
from ._comun import argumentos, carpeta, guardar_tabla, tabla_markdown

MONETARIAS = ["gasto_total", "ticket_medio", "ticket_mediano", "gasto_30d", "gasto_31_120d",
              "puntos_ganados", "puntos_canjeados"]


def escenarios(actual: pd.DataFrame, rng) -> dict[str, pd.DataFrame]:
    inflacion = actual.copy()
    inflacion[MONETARIAS] *= 1.3

    inactivos = actual.copy()
    inactivos["dias_desde_ultima"] += 30
    inactivos[["gasto_30d", "n_trans_30d"]] *= 0.5

    nuevos = actual.copy()
    bajos = actual[actual["n_transacciones"] <= actual["n_transacciones"].quantile(0.25)]
    reemplazar = rng.random(len(nuevos)) < 0.3
    nuevos.loc[reemplazar, :] = bajos.sample(int(reemplazar.sum()), replace=True,
                                             random_state=SEMILLA).to_numpy()

    falla = actual.copy()
    falla.loc[rng.random(len(falla)) < 0.4, "edad"] = np.nan

    return {"sin cambios": actual, "inflación 30%": inflacion, "clientes inactivos": inactivos,
            "más clientes nuevos": nuevos, "falla de datos": falla}


def falsas_alarmas(pob: pd.DataFrame, tamanos, repeticiones: int, rng) -> pd.DataFrame:
    """PSI entre dos muestras de la MISMA población: todo lo que se marque es falsa alarma."""
    filas = []
    for n in tamanos:
        if 2 * n > len(pob):
            continue
        psis = []
        for _ in range(repeticiones):
            idx = rng.permutation(len(pob))
            a, b = pob.iloc[idx[:n]], pob.iloc[idx[n:2 * n]]
            psis += [monitoreo.psi(a[v], b[v]) for v in VARIABLES_MODELO]
        psis = np.array(psis)
        filas.append({"clientes_por_muestra": n, "psi_medio": psis.mean(),
                      "psi_esperado": monitoreo.psi_esperado_sin_drift(n, n),
                      "frac_revisar": float((psis > monitoreo.UMBRAL_REVISAR).mean()),
                      "frac_alerta": float((psis > monitoreo.UMBRAL_ALERTA).mean())})
    return pd.DataFrame(filas)


def main(argv=None) -> None:
    args = argumentos(__doc__).parse_args(argv)
    destino = carpeta("etapa6")
    rng = np.random.default_rng(SEMILLA)

    pob = datos.cargar_poblacion()[VARIABLES_MODELO].reset_index(drop=True)
    mitad = rng.random(len(pob)) < 0.5
    referencia, actual = pob[mitad].reset_index(drop=True), pob[~mitad].reset_index(drop=True)
    print(f"Referencia: {len(referencia):,} clientes; actual: {len(actual):,}")

    # Modelo de uplift entrenado con el experimento, para ver el drift de las predicciones
    exp = datos.cargar_experimento()
    if args.rapido:
        exp = exp.sample(frac=0.3, random_state=SEMILLA)
    modelo = uplift.construir("t").fit(exp[VARIABLES_MODELO], exp["t"].to_numpy(), exp["y"].to_numpy())
    tau_ref = modelo.efecto(referencia)
    perfil = monitoreo.perfil(referencia, VARIABLES_MODELO)

    filas, mapa = [], {}
    for nombre, df in escenarios(actual, rng).items():
        problemas = monitoreo.validar(df, perfil)
        drift = monitoreo.reporte_drift(referencia, df, VARIABLES_MODELO)
        tau = modelo.efecto(df[VARIABLES_MODELO])
        mapa[nombre] = drift.set_index("variable")["psi"]
        filas.append({
            "escenario": nombre,
            "errores_validacion": int((problemas["nivel"] == "error").sum()),
            "avisos_validacion": int((problemas["nivel"] == "aviso").sum()),
            "detalle_validacion": "; ".join(problemas["variable"] + ": " + problemas["mensaje"]) or "-",
            "variables_en_alerta": int((drift["nivel"] == "alerta").sum()),
            "variables_a_revisar": int((drift["nivel"] == "revisar").sum()),
            "variable_mas_afectada": drift["variable"].iloc[0],
            "psi_max": float(drift["psi"].max()),
            "sliced_wasserstein": monitoreo.sliced_wasserstein(referencia, df[VARIABLES_MODELO]),
            "psi_prediccion": monitoreo.psi(tau_ref, tau),
            "cambio_tau_medio": float(np.mean(tau) - np.mean(tau_ref)),
        })
        print(f"  {nombre:<20} alertas: {filas[-1]['variables_en_alerta']}, PSI de la predicción: {filas[-1]['psi_prediccion']:.3f}")
    res = pd.DataFrame(filas)
    guardar_tabla(res, destino / "escenarios.csv")

    mapa = pd.DataFrame(mapa)
    mapa = mapa.loc[mapa.max(axis=1).sort_values(ascending=False).index]
    graficos.mapa_calor(mapa, destino / "psi.png", "PSI por variable y escenario",
                        f"PSI (color saturado en {monitoreo.UMBRAL_ALERTA})", tope=monitoreo.UMBRAL_ALERTA)

    print("Falsas alarmas según el tamaño de muestra...")
    fa = falsas_alarmas(pob, (200, 500, 2_000, 10_000, 50_000), 3 if args.rapido else 10, rng)
    guardar_tabla(fa, destino / "falsas_alarmas.csv", 5)

    texto = f"""# Etapa 6 — Validación y monitoreo de drift

Referencia: {len(referencia):,} clientes; datos "actuales": la otra mitad ({len(actual):,}), con cambios conocidos.
Umbrales de PSI: > {monitoreo.UMBRAL_REVISAR} revisar, > {monitoreo.UMBRAL_ALERTA} alerta.

## Escenarios

{tabla_markdown(res, 3)}

- `errores_validacion`: lo que haría frenar el pipeline antes de puntuar. La validación no distingue una falla de carga de un cambio real de la población (por ejemplo, más clientes nuevos implica más faltantes en las variables que necesitan varias compras): frena y deja la decisión a una persona.
- `psi_prediccion`: PSI de la distribución del uplift predicho. Un cambio en las variables que el modelo casi no usa puede no moverla; uno en las que usa, sí.

![PSI por variable y escenario](psi.png)

## Falsas alarmas: PSI entre dos muestras de la misma población

{tabla_markdown(fa, 4)}

Con muestras chicas, el PSI supera los umbrales habituales sin que haya ningún cambio: el umbral tiene que tener en cuenta el tamaño de muestra (el PSI esperado sin drift es aproximadamente (bins − 1)(1/n_ref + 1/n_act)).
"""
    (destino / "resumen.md").write_text(texto, encoding="utf-8")
    print("\n" + texto)
    print(f"Resultados en {destino}")


if __name__ == "__main__":
    main()