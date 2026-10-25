"""
Etapa 6: pipeline de producción.

    python -m src.pipeline entrenar [--modelo x]
    python -m src.pipeline puntuar [--version 20261025-153000] [--presupuesto 0.3 | optimo] [--forzar]
    python -m src.pipeline monitorear [--version ...]
    python -m src.pipeline versiones

entrenar    Entrena el modelo de uplift (por defecto el elegido en la etapa 4) y
            el de probabilidad de compra con TODO el experimento, y guarda una
            versión nueva en modelos/<version>/: el modelo, el perfil de los
            datos para validar, una muestra de referencia para el drift y la
            metadata (fecha, variables, tamaño, hash de los datos, métricas).
puntuar     Valida la población actual contra el perfil de entrenamiento (frena
            si hay errores, salvo --forzar), puntúa, arma la política con la
            mochila y escribe las tablas `puntajes` y `asignaciones` en Postgres.
            El presupuesto es una fracción del costo de tratar a todos, u
            "optimo": el que maximiza la ganancia neta esperada.
            Un grupo de control permanente (elegido con un hash del client_id)
            nunca recibe el incentivo: es lo que permite seguir midiendo el
            efecto real después de lanzar la política.
monitorear  Compara la población actual con la de entrenamiento (PSI y
            Wasserstein por variable, sliced Wasserstein y drift de las
            predicciones) y lo guarda en results/monitoreo/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from . import datos, db, monitoreo, politica, uplift
from .config import DIR_MODELOS, DIR_RESULTADOS, SEMILLA, VARIABLES_MODELO
from .politica import Supuestos

FRACCION_CONTROL = 0.10
SAL_CONTROL = "control-permanente:"     # independiente del hash de --muestra en src/carga.py
TAMANO_REFERENCIA = 20_000


# ─── Grupo de control permanente ─────────────────────────────────────────────
def en_control_permanente(client_id: str, fraccion: float = FRACCION_CONTROL) -> bool:
    """
    Determinístico: el mismo cliente queda siempre en el mismo grupo, corrida
    tras corrida. La sal hace que no se correlacione con la muestra de
    `--muestra` (que usa el mismo hash sin sal): sin ella, al cargar el 20% de
    los clientes, el control quedaría vacío o sería toda la muestra.
    """
    return zlib.crc32((SAL_CONTROL + str(client_id)).encode()) % 10_000 < fraccion * 10_000


# ─── Versiones ───────────────────────────────────────────────────────────────
def versiones() -> list[str]:
    if not DIR_MODELOS.exists():
        return []
    return sorted(p.name for p in DIR_MODELOS.iterdir() if (p / "metadata.json").exists())


def carpeta_version(version: str | None) -> Path:
    disponibles = versiones()
    if not disponibles:
        sys.exit("No hay modelos entrenados. Corré: python -m src.pipeline entrenar")
    version = version or disponibles[-1]
    if version not in disponibles:
        sys.exit(f"No existe la versión {version}. Disponibles: {', '.join(disponibles)}")
    return DIR_MODELOS / version


def cargar_version(version: str | None = None) -> tuple[dict, dict, pd.DataFrame]:
    carpeta = carpeta_version(version)
    metadata = json.loads((carpeta / "metadata.json").read_text(encoding="utf-8"))
    # pickle ejecuta código al cargar: solo se cargan modelos generados por este pipeline
    with open(carpeta / "modelos.pkl", "rb") as f:
        modelos = pickle.load(f)
    referencia = pd.read_csv(carpeta / "referencia.csv")
    return metadata, modelos, referencia


def _hash_datos(df: pd.DataFrame) -> str:
    contenido = pd.util.hash_pandas_object(df[["client_id", "t", "y"]], index=False).to_numpy().tobytes()
    return hashlib.sha256(contenido).hexdigest()[:16]


# ─── Comandos ────────────────────────────────────────────────────────────────
def entrenar(clave: str | None = None) -> str:
    elegido = DIR_RESULTADOS / "etapa4" / "modelo_elegido.txt"
    clave = clave or (elegido.read_text(encoding="utf-8").strip() if elegido.exists() else "x")
    df = datos.cargar_experimento()
    X, t, y = df[VARIABLES_MODELO], df["t"].to_numpy(), df["y"].to_numpy()
    print(f"Entrenando {uplift.MODELOS[clave].nombre} con {len(df):,} clientes del experimento...")
    modelo = uplift.construir(clave).fit(X, t, y)
    respuesta = uplift.TLearner().fit(X, t, y)

    version = datetime.now().strftime("%Y%m%d-%H%M%S")
    carpeta = DIR_MODELOS / version
    carpeta.mkdir(parents=True)
    with open(carpeta / "modelos.pkl", "wb") as f:
        pickle.dump({"uplift": modelo, "respuesta": respuesta}, f)

    poblacion = datos.cargar_poblacion()
    referencia = poblacion.sample(min(TAMANO_REFERENCIA, len(poblacion)), random_state=SEMILLA)
    referencia = referencia[["client_id", *VARIABLES_MODELO]].copy()
    referencia["tau"] = modelo.efecto(referencia[VARIABLES_MODELO])
    referencia.to_csv(carpeta / "referencia.csv", index=False)

    metricas = DIR_RESULTADOS / "etapa4" / "qini.csv"
    metadata = {
        "version": version,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "modelo": clave,
        "nombre_modelo": modelo.nombre,
        "variables": VARIABLES_MODELO,
        "clientes_entrenamiento": len(df),
        "tasa_tratados": float(t.mean()),
        "hash_datos": _hash_datos(df),
        "scikit_learn": sklearn.__version__,
        "perfil": monitoreo.perfil(df, VARIABLES_MODELO),
        "metricas_etapa4": pd.read_csv(metricas).to_dict(orient="records") if metricas.exists() else None,
    }
    (carpeta / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Versión {version} guardada en {carpeta}")
    return version


def puntuar(version: str | None, presupuesto: str | float, sup: Supuestos, forzar: bool = False) -> None:
    metadata, modelos, _ = cargar_version(version)
    version = metadata["version"]
    if metadata["variables"] != VARIABLES_MODELO:
        sys.exit("Las variables del modelo no coinciden con las de src/config.py: reentrená.")
    if metadata["scikit_learn"] != sklearn.__version__:
        print(f"AVISO: modelo entrenado con scikit-learn {metadata['scikit_learn']}, ahora {sklearn.__version__}")

    pob = datos.cargar_poblacion()
    print(f"Población: {len(pob):,} clientes. Validando datos...")
    problemas = monitoreo.validar(pob, metadata["perfil"])
    if len(problemas):
        print(problemas.to_string(index=False))
    if (problemas["nivel"] == "error").any() and not forzar:
        sys.exit("Validación con errores: no se puntúa (usá --forzar para ignorarlos bajo tu responsabilidad).")

    X = pob[VARIABLES_MODELO]
    tau = modelos["uplift"].efecto(X)
    mu0, _ = modelos["respuesta"].respuestas(X)
    mu1 = np.clip(mu0 + tau, 0, 1)
    monto = politica.monto_esperado(pob["ticket_medio"])
    v, c = politica.valor_y_costo(mu0, mu1, monto, sup)

    control = pob["client_id"].astype(str).map(en_control_permanente).to_numpy()
    elegibles = np.flatnonzero(~control)
    costo_todos = float(c[elegibles].sum())
    if str(presupuesto) == "optimo":
        B = politica.presupuesto_optimo(v[elegibles], c[elegibles])
    else:
        B = float(presupuesto) * costo_todos
    sol = politica.mochila(v[elegibles], c[elegibles], B)
    tratar = np.zeros(len(pob), dtype=bool)
    tratar[elegibles[sol.elegidos]] = True

    fecha = datetime.now().replace(microsecond=0)
    puntajes = pd.DataFrame({"client_id": pob["client_id"], "version": version, "fecha": fecha,
                             "tau": tau, "mu0": mu0, "mu1": mu1, "valor": v, "costo": c})
    asignaciones = pd.DataFrame({
        "client_id": pob["client_id"], "version": version, "fecha": fecha,
        "grupo": np.where(control, "control permanente", np.where(tratar, "tratar", "no tratar")),
        "tratar": tratar,
    })
    # Historial: se agregan filas; si se repuntúa la misma versión, se reemplaza
    for tabla, df in (("puntajes", puntajes), ("asignaciones", asignaciones)):
        if db.existe_tabla(tabla):
            db.ejecutar(f"DELETE FROM {tabla} WHERE version = %s", (version,))
        db.escribir_tabla(df, tabla, reemplazar=False)

    print(f"\nVersión {version}, presupuesto {B / costo_todos:.1%} del costo de tratar a los elegibles"
          f"{' (óptimo)' if str(presupuesto) == 'optimo' else ''}")
    print(f"  control permanente: {control.sum():,} ({control.mean():.1%})")
    print(f"  a tratar:           {tratar.sum():,} ({tratar.mean():.1%})")
    print(f"  gasto esperado:     {sol.costo:,.0f} de {B:,.0f}")
    print(f"  ganancia esperada:  {sol.valor - sol.costo:,.0f} (neta, según el modelo)")
    print(f"  precio sombra:      {sol.precio_sombra:.2f}")
    print("Tablas actualizadas: puntajes, asignaciones")


def monitorear(version: str | None) -> pd.DataFrame:
    metadata, modelos, referencia = cargar_version(version)
    pob = datos.cargar_poblacion()
    tabla = monitoreo.reporte_drift(referencia, pob, VARIABLES_MODELO)
    tau_actual = modelos["uplift"].efecto(pob[VARIABLES_MODELO])
    pred = {"variable": "PREDICCIÓN: tau", "psi": monitoreo.psi(referencia["tau"], tau_actual),
            "wasserstein": monitoreo.wasserstein_estandarizado(referencia["tau"], tau_actual),
            "nulos_ref": 0.0, "nulos_act": 0.0}
    pred["nivel"] = monitoreo.nivel_psi(pred["psi"])
    tabla = pd.concat([pd.DataFrame([pred]), tabla], ignore_index=True)
    sliced = monitoreo.sliced_wasserstein(referencia[VARIABLES_MODELO], pob[VARIABLES_MODELO])
    ruido = monitoreo.psi_esperado_sin_drift(len(referencia), len(pob))

    destino = DIR_RESULTADOS / "monitoreo"
    destino.mkdir(parents=True, exist_ok=True)
    archivo = destino / f"drift_{metadata['version']}_{datetime.now():%Y%m%d-%H%M%S}.csv"
    tabla.to_csv(archivo, index=False, float_format="%.5f")
    print(tabla.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSliced Wasserstein (todas las variables juntas): {sliced:.4f}")
    print(f"PSI esperado sin drift con estos tamaños: {ruido:.4f}")
    alertas = tabla[tabla["nivel"] == "alerta"]["variable"].tolist()
    print(f"Alertas: {', '.join(alertas) if alertas else 'ninguna'}")
    print(f"Reporte completo en {archivo}")
    return tabla


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="comando", required=True)
    e = sub.add_parser("entrenar")
    e.add_argument("--modelo", choices=list(uplift.MODELOS))
    s = sub.add_parser("puntuar")
    s.add_argument("--version")
    s.add_argument("--presupuesto", default="0.3", help='fracción del costo de tratar a todos, u "optimo"')
    s.add_argument("--margen", type=float, default=Supuestos.margen)
    s.add_argument("--descuento", type=float, default=Supuestos.descuento)
    s.add_argument("--contacto", type=float, default=Supuestos.costo_contacto)
    s.add_argument("--forzar", action="store_true")
    m = sub.add_parser("monitorear")
    m.add_argument("--version")
    sub.add_parser("versiones")
    args = p.parse_args(argv)

    if args.comando == "entrenar":
        entrenar(args.modelo)
    elif args.comando == "puntuar":
        puntuar(args.version, args.presupuesto, Supuestos(args.margen, args.descuento, args.contacto), args.forzar)
    elif args.comando == "monitorear":
        monitorear(args.version)
    else:
        for v in versiones():
            meta = json.loads((DIR_MODELOS / v / "metadata.json").read_text(encoding="utf-8"))
            print(f"{v}  {meta['nombre_modelo']:<22} {meta['clientes_entrenamiento']:>8,} clientes  datos {meta['hash_datos']}")


if __name__ == "__main__":
    main()