"""
Configuración del proyecto: rutas, conexión a la base y variables.

La conexión se lee de variables de entorno; si existe un archivo .env en la raíz,
sus valores se usan como respaldo (las variables de entorno tienen prioridad).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIR_DATOS = RAIZ / "data"
DIR_SQL = RAIZ / "sql"
DIR_RESULTADOS = RAIZ / "results"
DIR_MODELOS = RAIZ / "modelos"

SEMILLA = 2026


def _leer_dotenv(ruta: Path = RAIZ / ".env") -> None:
    """Carga CLAVE=valor del .env en os.environ sin pisar lo que ya esté definido."""
    if not ruta.exists():
        return
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, _, valor = linea.partition("=")
        os.environ.setdefault(clave.strip(), valor.strip().strip("\"'"))


_leer_dotenv()


@dataclass(frozen=True)
class ConexionDB:
    host: str
    puerto: int
    usuario: str
    clave: str
    base: str

    @classmethod
    def desde_entorno(cls) -> "ConexionDB":
        return cls(
            host=os.environ.get("DB_HOST", "localhost"),
            puerto=int(os.environ.get("DB_PUERTO", "5434")),
            usuario=os.environ.get("DB_USUARIO", "incentivos"),
            clave=os.environ.get("DB_CLAVE", "incentivos"),
            base=os.environ.get("DB_NOMBRE", "incentivos"),
        )

    def url_sqlalchemy(self) -> str:
        return f"postgresql+psycopg://{self.usuario}:{self.clave}@{self.host}:{self.puerto}/{self.base}"

    def parametros_psycopg(self) -> dict:
        return {
            "host": self.host,
            "port": self.puerto,
            "user": self.usuario,
            "password": self.clave,
            "dbname": self.base,
        }


# Variables por cliente (todas previas a la comunicación). Se usan para el
# chequeo de balance (etapa 2) y van a ser las entradas de los modelos de uplift
# (etapas 3 y 4). El género queda afuera a propósito: usarlo para decidir quién
# recibe un incentivo es discutible y en algunos países ilegal. Se conserva en la
# tabla solo para controlar que la política no lo discrimine de rebote.
VARIABLES_MODELO = [
    "n_transacciones",
    "gasto_total",
    "ticket_medio",
    "ticket_mediano",
    "dias_desde_ultima",
    "dias_desde_primera",
    "n_trans_30d",
    "gasto_30d",
    "gasto_31_120d",
    "tendencia_gasto",
    "intervalo_medio_dias",
    "intervalo_desvio_dias",
    "puntos_ganados",
    "puntos_canjeados",
    "frac_trans_con_canje",
    "n_tiendas",
    "n_productos",
    "unidades_por_transaccion",
    "edad",
    "antiguedad_dias",
    "dias_hasta_primer_canje",
    "canjeo_alguna_vez",
]

# Covariable previa al tratamiento para CUPED (reducción de varianza)
VARIABLE_PRE_TRATAMIENTO = "n_transacciones"