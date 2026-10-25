"""
Etapa 3 — Simulación: ¿qué método de uplift recupera mejor un efecto conocido?

    python -m src.estudios.simulacion --rapido     (2 réplicas, muestras chicas: ~1 min)
    python -m src.estudios.simulacion              (5 réplicas: ~10-15 min)
    python -m src.estudios.simulacion --replicas 10 --modelos t x dr

Usa las variables reales de los clientes y genera resultados con un efecto
individual conocido (ver src/simulacion.py). En cada escenario y réplica: se
simula, se separa 70/30, cada método se entrena con el 70% y se evalúa en el 30%.

Genera en results/etapa3/:
    metricas.csv       una fila por escenario × réplica × método
    resumen.csv        media e IC 95% entre réplicas
    pehe.png           error del efecto individual por escenario y método
    captura.png        captura de la política del 30% por escenario y método
    qini_base.png      curvas Qini del escenario base (primera réplica)
    resumen.md         todo lo anterior en texto, para el README
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pandas as pd
from scipy import stats

from .. import datos, evaluacion, graficos, uplift
from ..config import SEMILLA, VARIABLES_MODELO
from ..simulacion import ESCENARIOS, simular
from ._comun import argumentos, carpeta, guardar_tabla, tabla_markdown

FRACCION_PRUEBA = 0.3
FRACCION_POLITICA = 0.3
SIN_MODELO = "Sin modelo (efecto promedio)"

METRICAS = {
    "pehe": "PEHE (menor es mejor)",
    "captura": f"Captura de la política del {FRACCION_POLITICA:.0%} (mayor es mejor)",
    "qini": "Coeficiente Qini (mayor es mejor)",
    "qini_relativo": "Qini / Qini del oráculo",
    "sesgo_ate": "Sesgo del efecto promedio",
}


def evaluar(tau_real, tau_hat, y, t, qini_oraculo) -> dict:
    qini = evaluacion.coeficiente_qini(y, t, tau_hat)
    return {
        "pehe": evaluacion.pehe(tau_real, tau_hat),
        "captura": evaluacion.captura_politica(tau_real, tau_hat, FRACCION_POLITICA),
        "qini": qini,
        "qini_relativo": qini / qini_oraculo if qini_oraculo > 0 else np.nan,
        "sesgo_ate": float(np.mean(tau_hat) - np.mean(tau_real)),
    }


def una_replica(X_real, escenario, semilla, modelos, curvas=None) -> list[dict]:
    d = simular(X_real, escenario, semilla)
    rng = np.random.default_rng(semilla)
    prueba = rng.random(len(d.y)) < FRACCION_PRUEBA
    Xe, Xp = d.X[~prueba], d.X[prueba]
    te, ye = d.t[~prueba], d.y[~prueba]
    tp, yp, tau_p = d.t[prueba], d.y[prueba], d.tau[prueba]

    # El oráculo ordena por el efecto real: es el techo del Qini con estos (y, t)
    qini_oraculo = evaluacion.coeficiente_qini(yp, tp, tau_p)
    comun = {"escenario": escenario.nombre, "semilla": semilla, "n_prueba": int(prueba.sum()),
             "ate_real": float(tau_p.mean()), "desvio_tau_real": float(tau_p.std())}

    # Referencia: el mismo efecto para todos. Su PEHE es el desvío del efecto
    # real; un método que no mejora esto no está captando heterogeneidad.
    ate = evaluacion.diferencia_de_medias(ye, te)
    filas = [{**comun, "metodo": SIN_MODELO, "segundos": 0.0,
              **evaluar(tau_p, np.full(len(yp), ate), yp, tp, qini_oraculo)}]
    if curvas is not None:
        curvas["Oráculo (efecto real)"] = evaluacion.curva_qini(yp, tp, tau_p)

    for clave in modelos:
        modelo = uplift.construir(clave, semilla)
        t0 = time.time()
        tau_hat = modelo.fit(Xe, te, ye).efecto(Xp)
        filas.append({**comun, "metodo": modelo.nombre, "segundos": time.time() - t0,
                      **evaluar(tau_p, tau_hat, yp, tp, qini_oraculo)})
        if curvas is not None:
            curvas[modelo.nombre] = evaluacion.curva_qini(yp, tp, tau_hat)
    return filas


def resumir(metricas: pd.DataFrame) -> pd.DataFrame:
    """Media e IC 95% (t de Student) de cada métrica entre réplicas."""
    filas = []
    for (esc, met), g in metricas.groupby(["escenario", "metodo"], sort=False):
        fila = {"escenario": esc, "metodo": met, "replicas": len(g)}
        for m in METRICAS:
            v = g[m].dropna().to_numpy()
            media = float(v.mean()) if len(v) else np.nan
            mitad = float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
            fila |= {m: media, f"{m}_ic_inf": media - mitad, f"{m}_ic_sup": media + mitad}
        fila["segundos"] = float(g["segundos"].mean())
        filas.append(fila)
    return pd.DataFrame(filas)


def grafico(resumen: pd.DataFrame, metrica: str, ruta, titulo: str, referencia=None) -> None:
    graficos.puntos_por_grupo(
        resumen.rename(columns={"escenario": "grupo", "metodo": "serie", metrica: "media",
                                f"{metrica}_ic_inf": "ic_inf", f"{metrica}_ic_sup": "ic_sup"}),
        ruta, titulo, METRICAS[metrica] + ", media e IC 95% entre réplicas", referencia,
    )


def ganadores(resumen: pd.DataFrame) -> pd.DataFrame:
    modelos = resumen[resumen["metodo"] != SIN_MODELO]
    filas = []
    for esc, g in modelos.groupby("escenario", sort=False):
        ref = resumen[(resumen["escenario"] == esc) & (resumen["metodo"] == SIN_MODELO)].iloc[0]
        mejor_pehe = g.loc[g["pehe"].idxmin()]
        mejor_cap = g.loc[g["captura"].idxmax()]
        filas.append({
            "escenario": esc,
            "mejor en PEHE": mejor_pehe["metodo"],
            "PEHE": mejor_pehe["pehe"],
            "PEHE sin modelo": ref["pehe"],
            "mejor en captura": mejor_cap["metodo"],
            "captura": mejor_cap["captura"],
            "captura sin modelo": ref["captura"],
        })
    return pd.DataFrame(filas)


def main(argv=None) -> None:
    p = argumentos(__doc__)
    p.add_argument("--replicas", type=int, default=None, help="réplicas por escenario (por defecto 5; 2 con --rapido)")
    p.add_argument("--modelos", nargs="+", default=list(uplift.MODELOS), choices=list(uplift.MODELOS),
                   help="métodos a comparar: " + ", ".join(f"{k}={v.nombre}" for k, v in uplift.MODELOS.items()))
    args = p.parse_args(argv)
    replicas = args.replicas or (2 if args.rapido else 5)
    destino = carpeta("etapa3")

    # Variables reales de toda la población (no solo del experimento): el
    # simulador remuestrea clientes de acá.
    X_real = datos.cargar_poblacion()[VARIABLES_MODELO].reset_index(drop=True)
    print(f"Clientes disponibles para simular: {len(X_real):,}")

    escenarios = ESCENARIOS
    if args.rapido:
        escenarios = [dataclasses.replace(e, n=max(2_000, e.n // 4)) for e in ESCENARIOS]

    filas, curvas = [], {}
    t0 = time.time()
    for i, esc in enumerate(escenarios):
        for r in range(replicas):
            print(f"  {esc.nombre:<22} réplica {r + 1}/{replicas}  (n = {esc.n:,})", flush=True)
            semilla = SEMILLA + 1000 * i + r
            filas += una_replica(X_real, esc, semilla, args.modelos,
                                 curvas if (i == 0 and r == 0) else None)
    print(f"Listo en {time.time() - t0:.0f}s")

    metricas = pd.DataFrame(filas)
    resumen = resumir(metricas)
    guardar_tabla(metricas, destino / "metricas.csv", 5)
    guardar_tabla(resumen, destino / "resumen.csv", 5)
    grafico(resumen, "pehe", destino / "pehe.png", "Error en el efecto individual")
    grafico(resumen, "captura", destino / "captura.png",
            f"¿Cuánto del óptimo se gana tratando al {FRACCION_POLITICA:.0%} elegido por el modelo?", 1.0)
    graficos.curvas_qini(curvas, destino / "qini_base.png",
                         f"Curvas Qini, escenario {escenarios[0].nombre} (conjunto de prueba)")

    # ── Resumen ─────────────────────────────────────────────────────────────
    gan = ganadores(resumen)
    base = resumen[resumen["escenario"] == escenarios[0].nombre]
    filas_esc = "\n".join(
        f"| {e.nombre} | {e.n:,} | {e.efecto_promedio} | {e.heterogeneidad} | {e.frac_no_molestar:.0%} |"
        for e in escenarios
    )
    texto = f"""# Etapa 3 — Simulación con efecto conocido

{replicas} réplicas por escenario; cada método se entrena con el {1 - FRACCION_PRUEBA:.0%} y se evalúa en el {FRACCION_PRUEBA:.0%}.
Las variables son las reales de {len(X_real):,} clientes; el resultado se genera con un efecto individual conocido.

## Escenarios (se varía un factor por vez respecto del base)

| escenario | clientes | efecto promedio | heterogeneidad | no molestar |
|---|---|---|---|---|
{filas_esc}

## Escenario base

{tabla_markdown(base[['metodo', 'pehe', 'captura', 'qini_relativo', 'sesgo_ate', 'segundos']])}

- **PEHE**: error en el efecto de cada cliente. *{SIN_MODELO}* asigna a todos el mismo efecto: su PEHE es el desvío del efecto real.
- **Captura**: tratando al {FRACCION_POLITICA:.0%} con mayor uplift predicho, qué fracción se obtiene de lo que ganaría la mejor elección posible. Sin modelo, la elección es al azar.
- **Qini relativo**: coeficiente Qini (calculado con compras observadas, como en datos reales) sobre el del oráculo que ordena por el efecto real. Es mucho más ruidoso que el PEHE y la captura, que usan el efecto real: con pocos clientes de prueba, hasta el ordenamiento al azar puede salir bien. Es la única de las tres métricas disponible en la etapa 4.

## Mejor método por escenario

{tabla_markdown(gan)}

![PEHE por escenario](pehe.png)

![Captura por escenario](captura.png)

![Curvas Qini](qini_base.png)
"""
    (destino / "resumen.md").write_text(texto, encoding="utf-8")
    print("\n" + texto)
    print(f"Resultados en {destino}")


if __name__ == "__main__":
    main()