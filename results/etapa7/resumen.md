# Etapa 7 — Diseño del A/B de validación

**Pregunta:** ¿la política de la mochila gana más que la práctica actual (*Mayor probabilidad de compra*) con el mismo presupuesto?

## Diseño propuesto

- **Unidad:** cliente, asignado con un hash del `client_id` (estable entre corridas, como el control permanente de la etapa 6). Los clientes del control permanente quedan afuera.
- **Grupos:** A aplica la mochila, B aplica *Mayor probabilidad de compra*, cada uno con un presupuesto proporcional a su tamaño. El control permanente mide, además, el efecto absoluto de cada política.
- **Métrica principal:** ganancia neta por cliente en la ventana de la campaña (margen − descuentos − contactos), con supuestos: margen 25%, descuento 10%, contacto 1.
- **Métricas de control:** gasto total contra presupuesto en cada grupo; tasa de compra; composición por género de los tratados.
- **Análisis:** diferencia de medias con CUPED usando `gasto_total` previo a la campaña; un solo análisis al final del horizonte fijado (sin mirar antes); control de SRM antes de leer el resultado.

## Lo que dicen los datos del experimento

- Ganancia por cliente bajo la mochila: media 64.37, desvío 80.28 (30,136 clientes de prueba compatibles)
- Bajo *Mayor probabilidad de compra*: media 62.27, desvío 83.34 (30,263 clientes)
- Diferencia estimada (DR): **1.65** por cliente, IC 95% [0.97, 2.32]
- Correlación de la métrica con `gasto_total`: ρ ≈ 0.47, así que CUPED reduce la varianza un 22%

## Tamaño de muestra (α = 0.05, potencia 80%)

La última columna es la fracción de los 400,162 clientes cargados en la base que haría falta (los dos grupos juntos). Si cargaste una muestra con `--muestra`, la población real es más grande y la fracción, proporcionalmente menor.

| efecto_supuesto | delta_por_cliente | clientes_por_grupo | con_cuped | fraccion_clientes_cargados_con_cuped |
|---|---|---|---|---|
| Estimado (DR, etapa 5) | 1.6472 | 38738 | 30128 | 0.1506 |
| Extremo inferior del IC 95% | 0.9706 | 111568 | 86772 | 0.4337 |
| Mitad del estimado | 0.8236 | 154950 | 120511 | 0.6023 |

Con el efecto estimado hacen falta 38,738 clientes por grupo sin CUPED. El IC de la diferencia es ancho: planificar con el efecto puntual da un A/B con poca potencia si el efecto real está cerca del extremo inferior.

![Potencia](potencia.png)

La simulación remuestrea las ganancias observadas (asimétricas y con muchos ceros) y aplica un test t de Welch: si coincide con la fórmula, la aproximación normal es razonable para estos tamaños.

## Por qué no mirar el resultado antes de tiempo (A/A, α = 0.05)

| miradas | falsos_positivos |
|---|---|
| 1 | 0.045 |
| 2 | 0.084 |
| 5 | 0.144 |
| 10 | 0.192 |
| 20 | 0.245 |

Frenar en la primera mirada con p < 0.05 infla la tasa de falsos positivos. Si hace falta mirar antes, usar un método secuencial (por ejemplo, límites de O'Brien-Fleming o un test mSPRT) en lugar del test fijo.

## Sample ratio mismatch en el experimento de X5

Tratados 99,981, control 100,058: p-valor del test chi-cuadrado contra 50/50 = 0.863 (sin evidencia de desbalance en la asignación).
