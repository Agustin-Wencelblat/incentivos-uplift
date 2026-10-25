# A quién darle un incentivo, con presupuesto limitado

Darle un incentivo a quien iba a comprar igual es simplemente perder plata. La idea de este proyecto es estimar **a quién le cambia el comportamiento** un incentivo (uplift modeling). Asi, decidimos **a quién dárselo con un presupuesto fijo** para maximizar la ganancia incremental, **evaluamos esa política con los datos del experimento original** y la empaquetamos en un **pipeline con monitoreo**, como se operaría en producción.

**Datos:** [X5 RetailHero](https://ods.ai/competitions/x5-retailhero-uplift-modeling/data): ~200 mil clientes con un tratamiento (una comunicación) asignado mitad y mitad, y ~46 millones de líneas de compras previas a la campaña.

**Stack:** Python · PostgreSQL · SQL · pandas · scikit-learn · SciPy · Docker · GitHub Actions

## Estado

Dividi el proyecto en distintas etapas para estructurar el desarollo y testearlo parte por parte.

| Etapa | Qué hace | Estado |
|---|---|---|
| 1. Datos y variables | Carga de X5 en Postgres, tipos, variables por cliente en SQL, controles de calidad | ✅ Implementada |
| 2. El experimento | Balance entre grupos, efecto promedio, CUPED, efecto por segmento | ✅ Implementada y testeada |
| 3. Simulación | Qué método de uplift recupera mejor un efecto conocido | ✅ Implementada y testeada |
| 4. Uplift en datos reales | Modelos, curva Qini, calibración del uplift | ⬜ Por hacer: `src/uplift.py`, `src/evaluacion.py` |
| 5. Política con presupuesto | Mochila, precio sombra, evaluación off-policy | ⬜ Por hacer: `src/politica.py`, `src/ope.py` |
| 6. Producción y monitoreo | Pipeline versionado, validación de datos, drift | ⬜ Por hacer: `src/pipeline.py`, `src/monitoreo.py` |
| 7. A/B de validación | Diseño y potencia del experimento que validaría la política | ⬜ Por hacer |

## Resultados

*TODO*

## Decisiones de diseño

- **Uplift, no probabilidad de compra.** Un modelo de probabilidad de compra no distingue a quien compra *por* el incentivo de quien compra igual. El objetivo es $\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]$.
- **Fuga de información en la fecha de primer canje.** `clients.first_redeem_date` puede ser posterior a la comunicación, y un primer canje después de la campaña implica una compra durante la campaña, que es justo lo que se quiere predecir. Las variables solo cuentan el canje si ocurrió antes de la fecha de corte. (En los notebooks de ejemplo de scikit-uplift, esa fecha sin filtrar es la variable más importante del modelo: una señal típica de fuga.)
- **Corrección en el tiempo.** Todas las variables usan solo información anterior a la última compra registrada, previa a la comunicación.
- **El monto se agrega por ticket, no por línea.** En X5 hay una fila por producto y el total del ticket se repite en cada una. Sumar por línea multiplicaría el gasto por la cantidad de productos.
- **Cargar como texto y convertir en SQL.** Ningún valor raro rompe la carga, y la tabla `control_conversiones` registra cuántos valores no se pudieron convertir en cada columna.
- **Muestra por cliente con un hash.** `--muestra` elige clientes con un hash determinístico del `client_id`: la muestra es la misma en las tres tablas y en todas las corridas.
- **El umbral de balance depende del tamaño de muestra.** La regla de $|\mathrm{SMD}| < 0.1$ supone muestras grandes. Bajo asignación al azar, el SMD tiene un desvío aproximado de $\sqrt{1/n_T + 1/n_C}$, así que el estudio marca una variable solo si supera $\max(0.1,\ 3 \sqrt{1/n_T + 1/n_C})$. Con ~200 mil clientes manda el 0.1.
- **CUPED con una covariable previa al tratamiento.** Reduce la varianza en un factor $1 - \rho^2$ sin sesgar el efecto. Con una covariable afectada por el tratamiento, el ajuste sesgaría el resultado (hay un test que lo muestra).
- **Comparaciones múltiples.** El efecto por segmento reporta p-valores corregidos por Holm.
- **El género no entra al modelo.** Se conserva en la tabla solo para controlar más adelante que la política no lo discrimine de rebote.

## Limitaciones conocidas

- La documentación de X5 no dice explícitamente que la asignación haya sido al azar; la etapa 2 lo verifica con el balance entre grupos.
- El experimento de X5 es una sola campaña: el monitoreo de drift (etapa 6) se va a tener que demostrar con una población simulada.
- X5 no publica márgenes: los supuestos económicos de la etapa 5 van a ser inventados, documentados y con análisis de sensibilidad.
