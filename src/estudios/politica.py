"""
Etapa 5 — Política con presupuesto y evaluación off-policy.

    python -m src.estudios.politica --rapido
    python -m src.estudios.politica
    python -m src.estudios.politica --presupuesto 0.2 --margen 0.3 --descuento 0.05 --contacto 2

Requiere haber corrido la etapa 4 (usa results/etapa4/predicciones_prueba.csv:
predicciones de modelos entrenados SIN los clientes de prueba).

Con las predicciones de los clientes de prueba se arman varias políticas con el
mismo presupuesto y se evalúan con las compras que esos clientes hicieron en el
experimento (IPW y doblemente robusto, ver src/ope.py). El presupuesto se
expresa como fracción de lo que costaría tratar a todos.

Genera en results/etapa5/:
    politicas.csv, politicas.png   ganancia incremental de cada política (presupuesto principal)
    curva_presupuesto.csv/.png     ganancia incremental según el presupuesto
    precio_sombra.png              precio sombra según el presupuesto
    sensibilidad.csv               presupuesto óptimo y ganancia según margen y descuento
    genero.csv                     composición por género de los tratados (control)
    politicas_prueba.csv           a quién trata cada política (lo usa la etapa 7)
    supuestos.json                 supuestos usados (lo usa la etapa 7)
    resumen.md
"""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from .. import datos, graficos, ope, politica, uplift
from ..config import DIR_RESULTADOS, SEMILLA
from ..politica import Supuestos
from ._comun import argumentos, carpeta, guardar_tabla, tabla_markdown

POR_MIL = 1000      # los valores se reportan por cada 1000 clientes
MOCHILA = "Mochila (uplift / costo)"
NADIE = "A nadie"


def cargar_predicciones() -> tuple[pd.DataFrame, str]:
    origen = DIR_RESULTADOS / "etapa4"
    archivo = origen / "predicciones_prueba.csv"
    if not archivo.exists():
        raise SystemExit("Faltan las predicciones de la etapa 4. Corré primero: python -m src.estudios.uplift_real")
    pred = pd.read_csv(archivo)
    clave = (origen / "modelo_elegido.txt").read_text(encoding="utf-8").strip()
    extra = datos.cargar_experimento()[["client_id", "ticket_medio", "genero"]]
    pred = pred.merge(extra, on="client_id", how="left", validate="one_to_one")
    return pred, clave


def politicas_con_presupuesto(d: pd.DataFrame, v, c, presupuesto: float, tau) -> tuple[dict, politica.Solucion]:
    sol = politica.mochila(v, c, presupuesto)
    return {
        MOCHILA: sol.elegidos,
        "Mayor uplift (sin mirar el costo)": politica.por_puntaje(tau, c, presupuesto),
        "Mayor probabilidad de compra": politica.por_puntaje(d["mu0"], c, presupuesto),
        "Al azar": politica.al_azar(c, presupuesto),
        "A todos (sin presupuesto)": np.ones(len(d), dtype=bool),
        NADIE: np.zeros(len(d), dtype=bool),
    }, sol


def main(argv=None) -> None:
    p = argumentos(__doc__)
    p.add_argument("--presupuesto", type=float, default=0.3, help="fracción del costo de tratar a todos (por defecto 0.3)")
    p.add_argument("--margen", type=float, default=Supuestos.margen)
    p.add_argument("--descuento", type=float, default=Supuestos.descuento)
    p.add_argument("--contacto", type=float, default=Supuestos.costo_contacto)
    args = p.parse_args(argv)
    repeticiones = 200 if args.rapido else 1000
    destino = carpeta("etapa5")
    sup = Supuestos(args.margen, args.descuento, args.contacto)

    d, clave = cargar_predicciones()
    nombre_modelo = uplift.MODELOS[clave].nombre
    tau = d[f"tau_{clave}"].to_numpy()
    monto = politica.monto_esperado(d["ticket_medio"])
    t, y = d["t"].to_numpy(), d["y"].to_numpy()
    prob_t = float(t.mean())       # P(T = 1), conocida por diseño (~0.5)
    v, c = politica.valor_y_costo(d["mu0"], d["mu1"], monto, sup)
    r = ope.recompensa(y, t, monto, sup)
    q0, q1 = ope.modelo_recompensa(d["mu0"], d["mu1"], monto, sup)
    g = ope.gasto(y, t, monto, sup)
    g0, g1 = ope.modelo_gasto(d["mu1"], monto, sup)
    costo_todos = float(c.sum())
    print(f"Clientes de prueba: {len(d):,}; modelo: {nombre_modelo}; {sup}")

    # ── 1. Presupuesto principal ────────────────────────────────────────────
    B = args.presupuesto * costo_todos
    pols, sol = politicas_con_presupuesto(d, v, c, B, tau)
    res = ope.evaluar(r, t, pols, prob_t, q0, q1, referencia=NADIE, repeticiones=repeticiones)
    gasto_dr = {k: ope.valor_dr(g, t, pi, prob_t, g0, g1) for k, pi in pols.items()}
    res["gasto_dr"] = res["politica"].map(gasto_dr)
    res["gasto_esperado"] = [float(c[pi].sum()) / len(d) for pi in pols.values()]
    for col in ["valor_ipw", "valor_dr", "ic_inf", "ic_sup", "dif_vs_referencia", "dif_ic_inf", "dif_ic_sup",
                "error_estandar_ipw", "error_estandar_dr", "gasto_dr", "gasto_esperado"]:
        res[col] *= POR_MIL
    guardar_tabla(res, destino / "politicas.csv", 2)
    con_presupuesto = res[~res["politica"].isin([NADIE])]
    graficos.efectos_con_intervalo(
        con_presupuesto.drop(columns=["ic_inf", "ic_sup"]).rename(columns={"politica": "etiqueta", "dif_vs_referencia": "efecto",
                                        "dif_ic_inf": "ic_inf", "dif_ic_sup": "ic_sup"}),
        destino / "politicas.png",
        f"Ganancia neta incremental con presupuesto del {args.presupuesto:.0%}",
        etiqueta_x="Ganancia neta vs. no tratar a nadie, por cada 1000 clientes (DR, IC 95%)",
    )

    # Para la etapa 7: a quién trata cada política y con qué supuestos
    guardar_tabla(pd.DataFrame({"client_id": d["client_id"], **{k: v_.astype(int) for k, v_ in pols.items()}}),
                  destino / "politicas_prueba.csv")
    (destino / "supuestos.json").write_text(json.dumps(asdict(sup) | {"presupuesto": args.presupuesto}, indent=2),
                                            encoding="utf-8")

    # Greedy vs. solución entera exacta, en una submuestra (milp con todos tarda)
    rng = np.random.default_rng(SEMILLA)
    sub = rng.choice(len(d), size=min(2000, len(d)), replace=False)
    B_sub = args.presupuesto * float(c[sub].sum())
    greedy = politica.mochila(v[sub], c[sub], B_sub)
    exacta = politica.mochila_exacta(v[sub], c[sub], B_sub)
    brecha = 1 - greedy.valor / exacta.valor if exacta.valor > 0 else 0.0

    # ── 2. Curva según el presupuesto ───────────────────────────────────────
    print("Curva de presupuesto...")
    filas, sombras = [], []
    for f in np.round(np.arange(0.05, 1.0001, 0.05), 2):
        pols_f, sol_f = politicas_con_presupuesto(d, v, c, f * costo_todos, tau)
        pols_f = {k: pols_f[k] for k in [MOCHILA, "Mayor uplift (sin mirar el costo)",
                                          "Mayor probabilidad de compra", "Al azar", NADIE]}
        e = ope.evaluar(r, t, pols_f, prob_t, q0, q1, referencia=NADIE,
                        repeticiones=max(100, repeticiones // 5), semilla=SEMILLA + int(100 * f))
        for _, fila in e[e["politica"] != NADIE].iterrows():
            filas.append({"serie": fila["politica"], "fraccion": f, "valor": POR_MIL * fila["dif_vs_referencia"],
                          "ic_inf": POR_MIL * fila["dif_ic_inf"], "ic_sup": POR_MIL * fila["dif_ic_sup"]})
        sombras.append({"serie": "Precio sombra", "fraccion": f, "valor": sol_f.precio_sombra})
    curva = pd.DataFrame(filas)
    guardar_tabla(curva, destino / "curva_presupuesto.csv", 3)
    graficos.lineas(curva, "fraccion", destino / "curva_presupuesto.png",
                    "Ganancia neta incremental según el presupuesto",
                    "Presupuesto (fracción del costo de tratar a todos)",
                    "Ganancia vs. no tratar, por 1000 clientes (DR)", referencia_y=0)
    graficos.lineas(pd.DataFrame(sombras), "fraccion", destino / "precio_sombra.png",
                    "Precio sombra del presupuesto (mochila)",
                    "Presupuesto (fracción del costo de tratar a todos)",
                    "Ganancia incremental por peso adicional", banda=False, referencia_y=1)

    # Presupuesto óptimo según el modelo (no usa las compras de prueba: si se
    # eligiera mirando la curva DR, la estimación quedaría sesgada hacia arriba)
    B_opt = politica.presupuesto_optimo(v, c)
    pi_opt = politica.mochila(v, c, B_opt).elegidos
    ev_opt = ope.evaluar(r, t, {"opt": pi_opt, NADIE: pols[NADIE]}, prob_t, q0, q1, NADIE, repeticiones)
    opt = ev_opt.iloc[0]
    ganancia_modelo = POR_MIL * float((v - c)[pi_opt].sum()) / len(d)

    # ── 3. Sensibilidad a los supuestos ─────────────────────────────────────
    print("Sensibilidad...")
    filas = []
    for m in (0.15, 0.25, 0.35):
        for desc in (0.05, 0.10, 0.20):
            s = Supuestos(m, desc, sup.costo_contacto)
            v_s, c_s = politica.valor_y_costo(d["mu0"], d["mu1"], monto, s)
            r_s = ope.recompensa(y, t, monto, s)
            q0_s, q1_s = ope.modelo_recompensa(d["mu0"], d["mu1"], monto, s)
            B_s = politica.presupuesto_optimo(v_s, c_s)
            pi_s = politica.mochila(v_s, c_s, B_s).elegidos
            e = ope.evaluar(r_s, t, {"opt": pi_s, NADIE: np.zeros(len(d), dtype=bool)}, prob_t, q0_s, q1_s,
                            NADIE, max(100, repeticiones // 5)).iloc[0]
            filas.append({"margen": m, "descuento": desc, "tratados": float(pi_s.mean()),
                          "presupuesto_optimo": B_s / float(c_s.sum()),
                          "ganancia_esperada": POR_MIL * float((v_s - c_s)[pi_s].sum()) / len(d),
                          "ganancia_dr": POR_MIL * e["dif_vs_referencia"],
                          "ic_inf": POR_MIL * e["dif_ic_inf"], "ic_sup": POR_MIL * e["dif_ic_sup"]})
    sens = pd.DataFrame(filas)
    guardar_tabla(sens, destino / "sensibilidad.csv", 3)

    # ── 4. Control: composición por género de los elegidos ──────────────────
    genero = d["genero"].fillna("sin dato")
    comp = pd.DataFrame({
        "poblacion": genero.value_counts(normalize=True),
        "tratados_mochila": genero[pols[MOCHILA]].value_counts(normalize=True),
    }).fillna(0.0)
    comp["razon"] = comp["tratados_mochila"] / comp["poblacion"]
    comp = comp.reset_index(names="genero")
    guardar_tabla(comp, destino / "genero.csv", 3)

    # ── Resumen ─────────────────────────────────────────────────────────────
    fila_m = res[res["politica"] == MOCHILA].iloc[0]
    texto = f"""# Etapa 5 — Política con presupuesto

- Clientes de prueba: {len(d):,}. Uplift de **{nombre_modelo}** (elegido en la etapa 4); probabilidad sin incentivo del modelo de control del T-learner.
- Supuestos: margen {sup.margen:.0%}, descuento {sup.descuento:.0%} del ticket si el tratado compra, costo por contacto {sup.costo_contacto:g}. Monto de la compra: ticket medio del cliente.
- Valores en ganancia neta (margen − descuentos − contactos) por cada 1000 clientes, en la moneda de los tickets.

## Presupuesto del {args.presupuesto:.0%} del costo de tratar a todos

{tabla_markdown(res[['politica', 'tratados', 'dif_vs_referencia', 'dif_ic_inf', 'dif_ic_sup', 'gasto_esperado', 'gasto_dr']], 2)}

- `dif_vs_referencia`: ganancia neta frente a no tratar a nadie, estimador doblemente robusto, IC 95% bootstrap pareado.
- `gasto_esperado`: gasto que predice el modelo (con eso se arma la política); `gasto_dr`: gasto estimado con las compras observadas.
- Precio sombra con este presupuesto: **{sol.precio_sombra:.2f}**: un peso más de presupuesto agrega {sol.precio_sombra:.2f} de ganancia incremental{' (menos de lo que cuesta: conviene achicar el presupuesto)' if 0 < sol.precio_sombra < 1 else ''}.
- Greedy vs. solución entera exacta (milp, {len(sub):,} clientes): el greedy queda a {100 * brecha:.3f}% del óptimo.
- Errores estándar con la mochila: IPW {fila_m['error_estandar_ipw']:.2f}, DR {fila_m['error_estandar_dr']:.2f} por 1000 clientes.

![Políticas](politicas.png)

## ¿Cuánto presupuesto?

El presupuesto que maximiza la ganancia neta esperada trata mientras el precio sombra sea mayor que 1. Se elige con el modelo, sin mirar las compras de prueba.

- Presupuesto óptimo: {B_opt / costo_todos:.1%} del costo de tratar a todos ({100 * pi_opt.mean():.1f}% de los clientes)
- Ganancia neta que predice el modelo: {ganancia_modelo:.2f} por 1000 clientes
- Ganancia neta estimada con las compras observadas (DR): {POR_MIL * opt['dif_vs_referencia']:.2f} por 1000 clientes, IC 95% [{POR_MIL * opt['dif_ic_inf']:.2f}, {POR_MIL * opt['dif_ic_sup']:.2f}]

Si el modelo promete bastante más de lo que se observa, está exagerando el uplift de los clientes que elige (ver la pendiente de calibración de la etapa 4): la mochila elige justo a los clientes con el uplift más sobreestimado. Antes de usar la política conviene recalibrar el uplift o achicar el presupuesto.

![Curva de presupuesto](curva_presupuesto.png)

![Precio sombra](precio_sombra.png)

## Sensibilidad a los supuestos (presupuesto óptimo en cada caso)

{tabla_markdown(sens, 3)}

## Control: composición por género de los tratados

El género no entra al modelo; esto controla que la política no lo discrimine de rebote a través de otras variables.

{tabla_markdown(comp, 3)}
"""
    (destino / "resumen.md").write_text(texto, encoding="utf-8")
    print("\n" + texto)
    print(f"Resultados en {destino}")


if __name__ == "__main__":
    main()