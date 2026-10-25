"""
Etapa 7 — Diseño del A/B que validaría la política.

    python -m src.estudios.ab --rapido
    python -m src.estudios.ab
    python -m src.estudios.ab --comparar "Al azar"

Requiere las etapas 4 y 5. La evaluación off-policy de la etapa 5 es una
estimación; antes de lanzar la política a todos, se la compara en un A/B contra
la práctica actual (por defecto, darle el incentivo a los de mayor probabilidad
de compra, con el mismo presupuesto).

Con las compras del experimento se estima cómo se distribuiría la ganancia por
cliente bajo cada política (los clientes cuyo tratamiento coincide con el que
elegiría la política), y con eso:
    - cuántos clientes por grupo hacen falta, con y sin CUPED;
    - la curva de potencia, con la fórmula y por simulación;
    - cuánto sube la tasa de falsos positivos si se mira el resultado antes de tiempo;
    - un control de sample ratio mismatch sobre el experimento de X5.

Genera en results/etapa7/: tamanos.csv, potencia.csv, potencia.png, espiadas.csv, resumen.md
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import datos, db, graficos, ope, potencia
from ..config import DIR_RESULTADOS
from ..politica import Supuestos, monto_esperado
from ._comun import argumentos, carpeta, guardar_tabla, tabla_markdown

NUEVA = "Mochila (uplift / costo)"
COVARIABLE_CUPED = "gasto_total"      # gasto previo a la campaña


def media_desvio_ponderados(x, w) -> tuple[float, float]:
    media = float(np.average(x, weights=w))
    return media, float(np.sqrt(np.average((x - media) ** 2, weights=w)))


def correlacion_ponderada(x, z, w) -> float:
    ok = np.isfinite(z)
    x, z, w = x[ok], z[ok], w[ok]
    mx, mz = np.average(x, weights=w), np.average(z, weights=w)
    cov = np.average((x - mx) * (z - mz), weights=w)
    return float(cov / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((z - mz) ** 2, weights=w)))


def main(argv=None) -> None:
    p = argumentos(__doc__)
    p.add_argument("--comparar", default="Mayor probabilidad de compra",
                   help="política de comparación (una columna de results/etapa5/politicas_prueba.csv)")
    p.add_argument("--alfa", type=float, default=0.05)
    p.add_argument("--potencia", type=float, default=0.8)
    args = p.parse_args(argv)
    repeticiones = 100 if args.rapido else 500
    destino = carpeta("etapa7")

    e4, e5 = DIR_RESULTADOS / "etapa4", DIR_RESULTADOS / "etapa5"
    if not (e5 / "politicas_prueba.csv").exists():
        raise SystemExit("Faltan resultados de la etapa 5. Corré primero: python -m src.estudios.politica")
    supuestos = json.loads((e5 / "supuestos.json").read_text(encoding="utf-8"))
    sup = Supuestos(supuestos["margen"], supuestos["descuento"], supuestos["costo_contacto"])
    pred = pd.read_csv(e4 / "predicciones_prueba.csv")
    pols = pd.read_csv(e5 / "politicas_prueba.csv")
    if args.comparar not in pols:
        raise SystemExit(f"No existe la política '{args.comparar}'. Opciones: {', '.join(pols.columns[1:])}")
    extra = datos.cargar_experimento()[["client_id", "ticket_medio", COVARIABLE_CUPED]]
    d = pred.merge(pols, on="client_id", validate="one_to_one").merge(extra, on="client_id", validate="one_to_one")

    t, y = d["t"].to_numpy(), d["y"].to_numpy()
    monto = monto_esperado(d["ticket_medio"])
    r = ope.recompensa(y, t, monto, sup)
    prob_t = float(t.mean())
    w = np.where(t == 1, 1 / prob_t, 1 / (1 - prob_t))
    covariable = d[COVARIABLE_CUPED].to_numpy(dtype=float)

    # ── Distribución de la ganancia por cliente bajo cada política ──────────
    grupos = {}
    for nombre in (NUEVA, args.comparar):
        coincide = t == d[nombre].to_numpy()
        media, desvio = media_desvio_ponderados(r[coincide], w[coincide])
        rho = correlacion_ponderada(r[coincide], covariable[coincide], w[coincide])
        grupos[nombre] = {"muestra": r[coincide], "media": media, "desvio": desvio, "rho": rho,
                          "clientes": int(coincide.sum())}
    A, B = grupos[NUEVA], grupos[args.comparar]
    q0, q1 = ope.modelo_recompensa(d["mu0"], d["mu1"], monto, sup)
    ev = ope.evaluar(r, t, {NUEVA: d[NUEVA].to_numpy(), args.comparar: d[args.comparar].to_numpy()},
                     prob_t, q0, q1, referencia=args.comparar, repeticiones=repeticiones * 2)
    fila = ev[ev["politica"] == NUEVA].iloc[0]
    delta, delta_inf, delta_sup = fila["dif_vs_referencia"], fila["dif_ic_inf"], fila["dif_ic_sup"]
    rho = float(np.mean([A["rho"], B["rho"]]))
    sa_c, sb_c = potencia.desvio_cuped(A["desvio"], rho), potencia.desvio_cuped(B["desvio"], rho)
    poblacion = db.contar("variables_clientes")

    # ── 1. Tamaño de muestra ────────────────────────────────────────────────
    filas = []
    supuestos_delta = [("Estimado (DR, etapa 5)", delta), ("Mitad del estimado", delta / 2)]
    if delta_inf > 0:
        supuestos_delta.insert(1, ("Extremo inferior del IC 95%", delta_inf))
    for etiqueta, dlt in supuestos_delta:
        n = potencia.tamano_por_grupo(dlt, A["desvio"], B["desvio"], args.alfa, args.potencia)
        n_c = potencia.tamano_por_grupo(dlt, sa_c, sb_c, args.alfa, args.potencia)
        filas.append({"efecto_supuesto": etiqueta, "delta_por_cliente": dlt,
                      "clientes_por_grupo": n, "con_cuped": n_c,
                      "fraccion_poblacion_con_cuped": 2 * n_c / poblacion if np.isfinite(n_c) else np.inf})
    tamanos = pd.DataFrame(filas)
    guardar_tabla(tamanos, destino / "tamanos.csv", 4)

    # ── 2. Curva de potencia: fórmula vs. simulación ────────────────────────
    n_obj = tamanos["clientes_por_grupo"].iloc[0]
    if np.isfinite(n_obj) and n_obj > 0:
        grilla = np.unique(np.geomspace(max(50, n_obj / 8), n_obj * 2, 8).astype(int))
    else:
        grilla = np.unique(np.geomspace(100, 100_000, 8).astype(int))
    filas = []
    for n in grilla:
        filas.append({"serie": "Fórmula", "n": n, "valor": float(potencia.potencia_teorica(n, delta, A["desvio"], B["desvio"], args.alfa))})
        filas.append({"serie": "Fórmula con CUPED", "n": n, "valor": float(potencia.potencia_teorica(n, delta, sa_c, sb_c, args.alfa))})
        # La simulación remuestrea las ganancias observadas: incluye la asimetría y los ceros
        filas.append({"serie": "Simulación (remuestreo)", "n": n,
                      "valor": potencia.potencia_simulada(A["muestra"], B["muestra"] - B["media"] + A["media"] - delta,
                                                          n, repeticiones, args.alfa)})
    curva = pd.DataFrame(filas)
    guardar_tabla(curva, destino / "potencia.csv", 4)
    graficos.lineas(curva, "n", destino / "potencia.png", f"Potencia para detectar {delta:.2f} por cliente",
                    "Clientes por grupo", "Potencia", banda=False, referencia_y=args.potencia)

    # ── 3. Espiar el resultado ──────────────────────────────────────────────
    espiadas = pd.DataFrame([
        {"miradas": k, "falsos_positivos": potencia.falsos_positivos_espiando(2_000, k, 1000 if args.rapido else 4000, args.alfa)}
        for k in (1, 2, 5, 10, 20)
    ])
    guardar_tabla(espiadas, destino / "espiadas.csv", 4)

    # ── 4. SRM en el experimento de X5 ──────────────────────────────────────
    exp_t = datos.cargar_experimento()["t"]
    conteos = [int((exp_t == 1).sum()), int((exp_t == 0).sum())]
    p_srm = potencia.srm(conteos)

    n_texto = "∞" if not np.isfinite(n_obj) else f"{int(n_obj):,}"
    texto = f"""# Etapa 7 — Diseño del A/B de validación

**Pregunta:** ¿la política de la mochila gana más que la práctica actual (*{args.comparar}*) con el mismo presupuesto?

## Diseño propuesto

- **Unidad:** cliente, asignado con un hash del `client_id` (estable entre corridas, como el control permanente de la etapa 6). Los clientes del control permanente quedan afuera.
- **Grupos:** A aplica la mochila, B aplica *{args.comparar}*, cada uno con un presupuesto proporcional a su tamaño. El control permanente mide, además, el efecto absoluto de cada política.
- **Métrica principal:** ganancia neta por cliente en la ventana de la campaña (margen − descuentos − contactos), con supuestos: margen {sup.margen:.0%}, descuento {sup.descuento:.0%}, contacto {sup.costo_contacto:g}.
- **Métricas de control:** gasto total contra presupuesto en cada grupo; tasa de compra; composición por género de los tratados.
- **Análisis:** diferencia de medias con CUPED usando `{COVARIABLE_CUPED}` previo a la campaña; un solo análisis al final del horizonte fijado (sin mirar antes); control de SRM antes de leer el resultado.

## Lo que dicen los datos del experimento

- Ganancia por cliente bajo la mochila: media {A['media']:.2f}, desvío {A['desvio']:.2f} ({A['clientes']:,} clientes de prueba compatibles)
- Bajo *{args.comparar}*: media {B['media']:.2f}, desvío {B['desvio']:.2f} ({B['clientes']:,} clientes)
- Diferencia estimada (DR): **{delta:.2f}** por cliente, IC 95% [{delta_inf:.2f}, {delta_sup:.2f}]
- Correlación de la métrica con `{COVARIABLE_CUPED}`: ρ ≈ {rho:.2f}, así que CUPED reduce la varianza un {100 * rho ** 2:.0f}%

## Tamaño de muestra (α = {args.alfa}, potencia {args.potencia:.0%})

{tabla_markdown(tamanos, 4)}

Con el efecto estimado hacen falta {n_texto} clientes por grupo sin CUPED. El IC de la diferencia es ancho: planificar con el efecto puntual da un A/B con poca potencia si el efecto real está cerca del extremo inferior.{' Hay supuestos de efecto para los que no alcanza toda la población (fracción > 1): habría que juntar varias campañas, aceptar menos potencia o un efecto mínimo mayor.' if (tamanos['fraccion_poblacion_con_cuped'] > 1).any() else ''}{' Como el IC incluye al 0, la etapa 5 no alcanza para afirmar que la mochila supera a la práctica actual: justamente por eso hace falta el A/B.' if delta_inf <= 0 else ''}

![Potencia](potencia.png)

La simulación remuestrea las ganancias observadas (asimétricas y con muchos ceros) y aplica un test t de Welch: si coincide con la fórmula, la aproximación normal es razonable para estos tamaños.

## Por qué no mirar el resultado antes de tiempo (A/A, α = {args.alfa})

{tabla_markdown(espiadas, 3)}

Frenar en la primera mirada con p < {args.alfa} infla la tasa de falsos positivos. Si hace falta mirar antes, usar un método secuencial (por ejemplo, límites de O'Brien-Fleming o un test mSPRT) en lugar del test fijo.

## Sample ratio mismatch en el experimento de X5

Tratados {conteos[0]:,}, control {conteos[1]:,}: p-valor del test chi-cuadrado contra 50/50 = {p_srm:.3g}{' (sin evidencia de desbalance en la asignación)' if p_srm > 0.001 else ' (**posible falla de asignación o registro**)'}.
"""
    (destino / "resumen.md").write_text(texto, encoding="utf-8")
    print("\n" + texto)
    print(f"Resultados en {destino}")


if __name__ == "__main__":
    main()