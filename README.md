# A quién darle un incentivo, con presupuesto limitado

Darle un incentivo a quien iba a comprar igual es simplemente perder plata. La idea de este proyecto es estimar **a quién le cambia el comportamiento** un incentivo (uplift modeling). Asi, decidimos **a quién dárselo con un presupuesto fijo** para maximizar la ganancia incremental, **evaluamos esa política con los datos del experimento original** y la empaquetamos en un **pipeline con monitoreo**, como se operaría en producción.

**Datos:** [X5 RetailHero](https://ods.ai/competitions/x5-retailhero-uplift-modeling/data): ~200 mil clientes con un tratamiento (una comunicación) asignado mitad y mitad, y ~46 millones de líneas de compras previas a la campaña.

**Stack:** Python · PostgreSQL · SQL · pandas · scikit-learn · SciPy · Docker · GitHub Actions

## Estado

| Etapa | Qué hace | Código |
|---|---|---|
| 1. Datos y variables | Carga de X5 en Postgres, tipos, variables por cliente en SQL, controles de calidad | `sql/`, `src/carga.py`, `src/construir.py` |
| 2. El experimento | Balance entre grupos, efecto promedio, CUPED, efecto por segmento | `src/estudios/experimento.py` |
| 3. Simulación | Qué método de uplift recupera mejor un efecto conocido | `src/simulacion.py`, `src/uplift.py` |
| 4. Uplift en datos reales | Selección del modelo por validación cruzada, Qini con IC bootstrap, calibración por decil | `src/estudios/uplift_real.py` |
| 5. Política con presupuesto | Mochila, precio sombra, presupuesto óptimo, evaluación off-policy (IPW y DR), sensibilidad | `src/politica.py`, `src/ope.py` |
| 6. Producción y monitoreo | Pipeline versionado, validación de datos, control permanente, drift (PSI, Wasserstein) | `src/pipeline.py`, `src/monitoreo.py` |
| 7. A/B de validación | Tamaño de muestra, CUPED, potencia simulada, costo de espiar, SRM | `src/potencia.py`, `src/estudios/ab.py` |

Las siete etapas están implementadas y testeadas (`pytest`: 41 tests).

## Resultados

Con el dataset completo de X5: 200.039 clientes en el experimento y 400.162 en total. Las ganancias están en rublos y dependen de supuestos económicos inventados (margen 25%, descuento del 10% del ticket si el tratado compra, 1 rublo por contacto), con análisis de sensibilidad. Los detalles están en los `resumen.md` de `results/`.

**1. El experimento fue al azar y la comunicación funciona** (etapa 2). Ninguna variable supera una diferencia estandarizada de 0,01 entre grupos y un clasificador no logra predecir el tratamiento (AUC 0,499). La tasa de compra pasa de 60,3% a 63,7%: $\hat\tau = 3{,}3$ puntos, IC 95% $[2{,}9;\ 3{,}8]$. El efecto cae con la frecuencia de compra: 4,3 puntos en el cuartil menos frecuente contra 1,4 en el más frecuente ($z \approx 5{,}8$ para la diferencia), y es mayor en quienes compraron hace más tiempo (4,1 contra 2,6 puntos). Quien compra seguido compra igual.

**2. En simulación, ordenar bien no es lo mismo que estimar bien** (etapa 3). Con las variables reales y un efecto conocido, el S-learner es el que mejor elige a quién tratar: tratando al 30% que elige, obtiene el 86% de lo que ganaría la mejor elección posible (al azar, el 44%). Pero subestima el efecto promedio en un 45%: ordena bien con valores achicados. Sin heterogeneidad real, cualquier modelo tiene más error que asignarle a todos el efecto promedio.

**3. En los datos reales, el modelo encuentra un grupo muy sensible, pero exagera** (etapa 4). El X-learner, elegido por validación cruzada, ordena mejor que el azar (Qini 0,0035, IC 95% $[0{,}0014;\ 0{,}0059]$), aunque en prueba no se distingue de los otros modelos. El decil de mayor uplift predicho responde 8,2 puntos, casi el triple del promedio; entre los deciles 2 y 10 casi no hay diferencias. La pendiente de calibración es 0,56: el modelo exagera las diferencias entre clientes. No aparece un grupo al que el incentivo le juegue en contra: el decil más bajo responde +3,0 puntos.

**4. Con un descuento del 10%, ningún presupuesto se paga** (etapa 5). El descuento se paga también en las compras que iban a ocurrir igual. Tratar a un cliente conviene si

$$\tau_i \cdot \text{margen} > \mu_{1,i} \cdot \text{descuento} \quad\Longleftrightarrow\quad \tau_i > 0{,}4\, \mu_{1,i} \approx 0{,}25,$$

y el mejor decil llega a 0,08. Con un presupuesto del 30% del costo de tratar a todos, todas las políticas pierden plata frente a no tratar a nadie: la mochila, 7,4 rublos por cliente; tratar a los de mayor probabilidad de compra, 9,1. El precio sombra es 0,15: cada rublo extra de presupuesto devuelve 15 centavos. El presupuesto óptimo trata al 0,4% de los clientes, con una ganancia indistinguible de cero (15,6 rublos cada 1000 clientes, IC $[-11;\ 45]$). Solo con margen 35% y descuento 5% el resultado se acerca a ser positivo. El problema no es a quién darle el incentivo sino cómo está diseñado: un descuento sobre toda la compra no se paga solo.

**5. Con el mismo presupuesto, la mochila pierde menos que la heurística habitual** (etapas 5 y 7). Frente a tratar a los de mayor probabilidad de compra, gana 1,65 rublos más por cliente (IC 95% $[0{,}97;\ 2{,}32]$). Confirmarlo con un A/B requiere unos 38.700 clientes por grupo, o 30.100 con CUPED (el 15% de la población). Mirar el resultado 10 veces antes de tiempo subiría los falsos positivos del 5% al 19%.

**6. El monitoreo detecta lo que importa** (etapa 6). Un mes más de inactividad de los clientes dispara alertas (PSI de la recencia: 8,3) y mueve la distribución del uplift predicho (PSI 0,23). Una falla de carga (40% de edades vacías) frena el pipeline antes de puntuar. Una inflación del 30% solo pide revisar dos variables de monto y casi no mueve las predicciones (PSI 0,002). Con 200 clientes por muestra, el 34% de las variables pide "revisar" sin ningún cambio real.



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
- **Simulación semisintética antes que datos reales.** En los datos reales nunca se observa el efecto de cada cliente, así que no hay forma directa de saber qué método lo estima mejor. La etapa 3 conserva las variables reales (con sus correlaciones, faltantes y colas) y solo inventa la respuesta, con un efecto conocido. Ahí se pueden medir el PEHE, $\sqrt{\mathbb{E}[(\tau(X) - \hat\tau(X))^2]}$, y la captura de la política, que en datos reales son imposibles.
- **La métrica que importa es el orden, no el valor.** Para decidir a quién darle el incentivo alcanza con ordenar bien a los clientes. Por eso se reporta, además del PEHE, qué parte del óptimo se gana tratando al 30% elegido por el modelo. La referencia *sin modelo* (el mismo efecto para todos) es el piso: un método que no la supera no está captando heterogeneidad.
- **Modelos base regularizados.** Los pseudo-resultados del X-, DR-learner y la variable transformada son muy ruidosos; con árboles chicos, hojas de al menos 100 clientes y parada temprana, el error del efecto individual baja a casi la mitad con muestras chicas.
- **El modelo se elige con validación cruzada, no con la prueba.** Si se eligiera el mejor Qini de prueba y después se reportara ese mismo Qini, el resultado estaría inflado por la propia elección.
- **Intervalos pareados.** Los modelos y las políticas se comparan sobre las mismas remuestras bootstrap: la incertidumbre común se cancela y el intervalo de la diferencia es mucho más angosto que comparar intervalos por separado.
- **El descuento se paga también en las compras que iban a ocurrir igual.** Por eso la mochila ordena por ganancia incremental sobre gasto esperado ($v_i / c_i$) y no solo por uplift: a un cliente fiel con uplift positivo le cuesta mucho más el descuento que lo que suma.
- **Precio sombra y presupuesto óptimo.** El precio sombra $\lambda$ es la ganancia incremental de un peso más de presupuesto. Si $\lambda < 1$, el último peso devuelve menos de lo que cuesta. El presupuesto óptimo es el que lleva $\lambda$ a 1, y se elige con el modelo, sin mirar las compras de prueba.
- **Evaluación off-policy doblemente robusta.** IPW es insesgado porque la asignación fue al azar, pero solo usa a los clientes cuyo tratamiento coincide con la política. El estimador DR agrega el modelo de recompensa y reduce la varianza sin perder el insesgamiento (hay un test que lo verifica con un modelo deliberadamente imperfecto).
- **Control permanente con sal.** El control usa un hash del `client_id` con una sal distinta a la de `--muestra`; sin ella, al cargar una muestra quedaría vacío o sería toda la muestra.
- **Umbrales de drift según el tamaño de muestra.** Sin drift, el PSI esperado es aproximadamente $(b - 1)(1/n_{ref} + 1/n_{act})$, así que con muestras chicas el umbral de 0.1 da falsas alarmas.
- **El género no entra al modelo.** Se conserva en la tabla solo para controlar más adelante que la política no lo discrimine de rebote.

## Limitaciones conocidas

- La documentación de X5 no dice explícitamente que la asignación haya sido al azar; la etapa 2 lo verifica con el balance entre grupos.
- X5 no publica márgenes ni costos: los supuestos económicos de la etapa 5 son inventados, por eso se reporta la sensibilidad a ellos.
- El resultado del experimento es si el cliente compró, no cuánto: el monto de la compra se aproxima con su ticket medio.
- El experimento es una sola campaña: el drift de la etapa 6 se demuestra con cambios simulados sobre la población real.
- La evaluación de la etapa 5 usa solo el 30% de prueba del experimento: los intervalos son anchos, y por eso la etapa 7 propone validar la política con un A/B antes de lanzarla.

## Posibles extensiones

- Recalibrar el uplift (por ejemplo, con la pendiente de la calibración por decil) antes de armar la política.
- Causal forest (`econml`) como sexto modelo.
- Ajuste de hiperparámetros con Qini por validación cruzada.
- Análisis secuencial del A/B (O'Brien-Fleming o mSPRT).
- Un incentivo cuyo costo caiga solo sobre compras incrementales (por ejemplo, un bono que se cobra solo si el cliente supera su gasto habitual), que es lo que muestra la etapa 5 como problema de fondo.
