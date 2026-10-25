"""
Etapa 4 — Uplift en el experimento real.

    python -m src.estudios.uplift_real --rapido     (30% de los clientes, 2 pliegues)
    python -m src.estudios.uplift_real
    python -m src.estudios.uplift_real --modelos t x dr

Pasos:
  1. Se separa el experimento en entrenamiento (70%) y prueba (30%).
  2. Selección del modelo con validación cruzada SOBRE ENTRENAMIENTO, por
     coeficiente Qini. Elegir con la prueba y después evaluar con la misma
     prueba sobreestimaría el resultado del modelo elegido.
  3. Cada modelo se reentrena con todo el entrenamiento y se evalúa en prueba:
     curvas Qini, coeficiente Qini con IC bootstrap pareado y calibración por
     decil del modelo elegido.
  4. Se guardan las predicciones de prueba para la etapa 5.

Genera en results/etapa4/:
    seleccion.csv            Qini por pliegue de validación cruzada
    qini.csv, qini.png       coeficiente Qini en prueba con IC 95%
    curvas_qini.png          curvas Qini en prueba
    calibracion.csv/.png     efecto predicho vs. observado por decil (modelo elegido)
    predicciones_prueba.csv  client_id, t, y, mu0, mu1 y el uplift de cada modelo
    modelo_elegido.txt       clave del modelo elegido
    resumen.md
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .. import datos, evaluacion, graficos, uplift
from ..config import SEMILLA, VARIABLES_MODELO
from ._comun import argumentos, carpeta, guardar_tabla, tabla_markdown

FRACCION_PRUEBA = 0.3


def separar(n: int, semilla: int = SEMILLA) -> np.ndarray:
    """Máscara de prueba, determinística."""
    return np.random.default_rng(semilla).random(n) < FRACCION_PRUEBA


def seleccion_por_validacion_cruzada(X, t, y, modelos, pliegues: int) -> pd.DataFrame:
    filas = []
    divisor = KFold(pliegues, shuffle=True, random_state=SEMILLA)
    for k, (entrena, valida) in enumerate(divisor.split(X), start=1):
        for clave in modelos:
            m = uplift.construir(clave).fit(X.iloc[entrena], t[entrena], y[entrena])
            q = evaluacion.coeficiente_qini(y[valida], t[valida], m.efecto(X.iloc[valida]))
            filas.append({"pliegue": k, "clave": clave, "metodo": m.nombre, "qini": q})
            print(f"    pliegue {k}: {m.nombre:<22} Qini = {q:.5f}", flush=True)
    return pd.DataFrame(filas)


def main(argv=None) -> None:
    p = argumentos(__doc__)
    p.add_argument("--modelos", nargs="+", default=list(uplift.MODELOS), choices=list(uplift.MODELOS))
    p.add_argument("--pliegues", type=int, default=None, help="pliegues de validación cruzada (por defecto 3; 2 con --rapido)")
    args = p.parse_args(argv)
    pliegues = args.pliegues or (2 if args.rapido else 3)
    repeticiones = 100 if args.rapido else 500
    destino = carpeta("etapa4")

    df = datos.cargar_experimento()
    if args.rapido:
        df = df.sample(frac=0.3, random_state=SEMILLA).reset_index(drop=True)
    X, t, y = df[VARIABLES_MODELO], df["t"].to_numpy(), df["y"].to_numpy()
    prueba = separar(len(df))
    Xe, te, ye = X[~prueba], t[~prueba], y[~prueba]
    Xp, tp, yp = X[prueba], t[prueba], y[prueba]
    print(f"Clientes: {len(df):,} (entrenamiento {(~prueba).sum():,}, prueba {prueba.sum():,})")

    # ── 1. Selección con validación cruzada en entrenamiento ────────────────
    print(f"Selección del modelo ({pliegues} pliegues sobre entrenamiento)...")
    sel = seleccion_por_validacion_cruzada(Xe, te, ye, args.modelos, pliegues)
    guardar_tabla(sel, destino / "seleccion.csv", 6)
    media_cv = sel.groupby(["clave", "metodo"], sort=False)["qini"].agg(["mean", "std"]).reset_index()
    elegido = media_cv.loc[media_cv["mean"].idxmax(), "clave"]
    nombre_elegido = uplift.MODELOS[elegido].nombre
    print(f"  Elegido: {nombre_elegido}")

    # ── 2. Reentrenar con todo el entrenamiento y evaluar en prueba ─────────
    print("Entrenamiento final y evaluación en prueba...")
    puntajes, segundos = {}, {}
    for clave in args.modelos:
        t0 = time.time()
        puntajes[uplift.MODELOS[clave].nombre] = uplift.construir(clave).fit(Xe, te, ye).efecto(Xp)
        segundos[uplift.MODELOS[clave].nombre] = time.time() - t0
    # Probabilidad de compra sin incentivo (mu0) para la etapa 5: el modelo de
    # control del T-learner, que solo ve clientes no tratados
    mu0, _ = uplift.TLearner().fit(Xe, te, ye).respuestas(Xp)

    print(f"Qini con IC bootstrap ({repeticiones} remuestras pareadas)...")
    boot = evaluacion.qini_bootstrap(yp, tp, puntajes, repeticiones)
    filas = []
    for nombre, tau_hat in puntajes.items():
        r = evaluacion.resumir_bootstrap(boot[nombre], evaluacion.coeficiente_qini(yp, tp, tau_hat))
        dif = boot[nombre] - boot[nombre_elegido]
        filas.append({
            "metodo": nombre, "qini": r["estimacion"], "ic_inf": r["ic_inf"], "ic_sup": r["ic_sup"],
            "prob_qini_positivo": r["prob_positivo"],
            "dif_vs_elegido_ic_inf": float(dif.quantile(0.025)), "dif_vs_elegido_ic_sup": float(dif.quantile(0.975)),
            "qini_cv": float(media_cv.loc[media_cv["metodo"] == nombre, "mean"].iloc[0]),
            "segundos": segundos[nombre],
        })
    qini = pd.DataFrame(filas)
    guardar_tabla(qini, destino / "qini.csv", 6)
    graficos.efectos_con_intervalo(
        qini.rename(columns={"metodo": "etiqueta", "qini": "efecto"}), destino / "qini.png",
        "Coeficiente Qini en prueba", etiqueta_x="Coeficiente Qini (0 = no ordena mejor que el azar), IC 95% bootstrap",
    )
    graficos.curvas_qini({k: evaluacion.curva_qini(yp, tp, v) for k, v in puntajes.items()},
                         destino / "curvas_qini.png", "Curvas Qini en prueba")

    # ── 3. Calibración del modelo elegido ───────────────────────────────────
    cal = evaluacion.calibracion_por_grupo(yp, tp, puntajes[nombre_elegido], grupos=10)
    guardar_tabla(cal, destino / "calibracion.csv", 5)
    graficos.calibracion(cal, destino / "calibracion.png", f"Calibración por decil: {nombre_elegido}")
    # Pendiente de la recta efecto observado ~ predicho: 1 = calibrado,
    # < 1 = el modelo exagera las diferencias entre clientes
    pendiente = float(np.polyfit(cal["uplift_predicho"], cal["efecto_observado"], 1, w=cal["clientes"])[0])
    top, fondo = cal.iloc[0], cal.iloc[-1]

    # ── 4. Predicciones para la etapa 5 ─────────────────────────────────────
    pred = pd.DataFrame({"client_id": df.loc[prueba, "client_id"].to_numpy(), "t": tp, "y": yp, "mu0": mu0})
    pred["mu1"] = np.clip(mu0 + puntajes[nombre_elegido], 0, 1)
    for clave in args.modelos:
        pred[f"tau_{clave}"] = puntajes[uplift.MODELOS[clave].nombre]
    guardar_tabla(pred, destino / "predicciones_prueba.csv", 6)
    (destino / "modelo_elegido.txt").write_text(elegido, encoding="utf-8")

    # ── Resumen ─────────────────────────────────────────────────────────────
    ate = evaluacion.efecto_promedio(yp, tp)
    fila_eleg = qini[qini["metodo"] == nombre_elegido].iloc[0]
    texto = f"""# Etapa 4 — Uplift en el experimento real

- Clientes: {len(df):,}; entrenamiento {(~prueba).sum():,}, prueba {prueba.sum():,}
- Efecto promedio en prueba: {ate['efecto']:.4f} [{ate['ic_inf']:.4f}, {ate['ic_sup']:.4f}]
- Modelo elegido por Qini en validación cruzada ({pliegues} pliegues, solo entrenamiento): **{nombre_elegido}**

## Coeficiente Qini en prueba (IC 95% bootstrap, {repeticiones} remuestras pareadas)

{tabla_markdown(qini[['metodo', 'qini_cv', 'qini', 'ic_inf', 'ic_sup', 'prob_qini_positivo', 'dif_vs_elegido_ic_inf', 'dif_vs_elegido_ic_sup']], 5)}

- `qini_cv`: media en validación cruzada (la que decide la elección). `qini`: en prueba.
- `dif_vs_elegido`: intervalo de la diferencia con el modelo elegido sobre las mismas remuestras. Si contiene al 0, la prueba no distingue a ese modelo del elegido.
- El modelo elegido ordena mejor que el azar con probabilidad bootstrap {fila_eleg['prob_qini_positivo']:.2f}.

![Curvas Qini](curvas_qini.png)

## Calibración por decil ({nombre_elegido})

- Decil de mayor uplift predicho: predicho {top['uplift_predicho']:.4f}, observado {top['efecto_observado']:.4f} [{top['ic_inf']:.4f}, {top['ic_sup']:.4f}]
- Decil de menor uplift predicho: predicho {fondo['uplift_predicho']:.4f}, observado {fondo['efecto_observado']:.4f} [{fondo['ic_inf']:.4f}, {fondo['ic_sup']:.4f}]
- Pendiente de observado sobre predicho: {pendiente:.2f} (1 = calibrado; menor que 1 = el modelo exagera las diferencias entre clientes)

{tabla_markdown(cal, 4)}

![Calibración](calibracion.png)
"""
    (destino / "resumen.md").write_text(texto, encoding="utf-8")
    print("\n" + texto)
    print(f"Resultados en {destino}")


if __name__ == "__main__":
    main()