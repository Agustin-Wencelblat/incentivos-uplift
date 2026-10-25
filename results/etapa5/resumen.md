# Etapa 5 — Política con presupuesto

- Clientes de prueba: 12,066. Uplift de **X-learner** (elegido en la etapa 4); probabilidad sin incentivo del modelo de control del T-learner.
- Supuestos: margen 25%, descuento 10% del ticket si el tratado compra, costo por contacto 1. Monto de la compra: ticket medio del cliente.
- Valores en ganancia neta (margen − descuentos − contactos) por cada 1000 clientes, en la moneda de los tickets.

## Presupuesto del 30% del costo de tratar a todos

| politica | tratados | dif_vs_referencia | dif_ic_inf | dif_ic_sup | gasto_esperado | gasto_dr |
|---|---|---|---|---|---|---|
| Mochila (uplift / costo) | 0.31 | -6714.77 | -7997.11 | -5289.67 | 9273.25 | 9172.32 |
| Mayor uplift (sin mirar el costo) | 0.27 | -6451.61 | -7687.80 | -5140.70 | 9273.25 | 9274.66 |
| Mayor probabilidad de compra | 0.25 | -8445.39 | -9094.43 | -7801.87 | 9273.30 | 9248.89 |
| Al azar | 0.30 | -7249.88 | -8392.30 | -6084.68 | 9273.27 | 9342.40 |
| A todos (sin presupuesto) | 1.00 | -25322.75 | -27280.19 | -23259.57 | 30911.05 | 30928.79 |
| A nadie | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

- `dif_vs_referencia`: ganancia neta frente a no tratar a nadie, estimador doblemente robusto, IC 95% bootstrap pareado.
- `gasto_esperado`: gasto que predice el modelo (con eso se arma la política); `gasto_dr`: gasto estimado con las compras observadas.
- Precio sombra con este presupuesto: **0.16**: un peso más de presupuesto agrega 0.16 de ganancia incremental (menos de lo que cuesta: conviene achicar el presupuesto).
- Greedy vs. solución entera exacta (milp, 2,000 clientes): el greedy queda a -0.002% del óptimo.
- Errores estándar con la mochila: IPW 1143.27, DR 876.08 por 1000 clientes.

![Políticas](politicas.png)

## ¿Cuánto presupuesto?

El presupuesto que maximiza la ganancia neta esperada trata mientras el precio sombra sea mayor que 1. Se elige con el modelo, sin mirar las compras de prueba.

- Presupuesto óptimo: 0.3% del costo de tratar a todos (0.7% de los clientes)
- Ganancia neta que predice el modelo: 16.68 por 1000 clientes
- Ganancia neta estimada con las compras observadas (DR): 38.19 por 1000 clientes, IC 95% [-21.47, 102.26]

Si el modelo promete bastante más de lo que se observa, está exagerando el uplift de los clientes que elige (ver la pendiente de calibración de la etapa 4): la mochila elige justo a los clientes con el uplift más sobreestimado. Antes de usar la política conviene recalibrar el uplift o achicar el presupuesto.

![Curva de presupuesto](curva_presupuesto.png)

![Precio sombra](precio_sombra.png)

## Sensibilidad a los supuestos (presupuesto óptimo en cada caso)

| margen | descuento | tratados | presupuesto_optimo | ganancia_esperada | ganancia_dr | ic_inf | ic_sup |
|---|---|---|---|---|---|---|---|
| 0.150 | 0.050 | 0.010 | 0.005 | 17.182 | 29.733 | -27.695 | 94.144 |
| 0.150 | 0.100 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.150 | 0.200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.250 | 0.050 | 0.045 | 0.031 | 172.032 | -275.046 | -728.518 | 216.268 |
| 0.250 | 0.100 | 0.007 | 0.003 | 16.683 | 38.189 | -12.081 | 89.744 |
| 0.250 | 0.200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.350 | 0.050 | 0.094 | 0.073 | 547.967 | -392.306 | -1619.485 | 671.143 |
| 0.350 | 0.100 | 0.019 | 0.010 | 93.575 | 12.768 | -313.661 | 290.208 |
| 0.350 | 0.200 | 0.001 | 0.001 | 1.562 | 20.602 | -11.913 | 50.031 |

## Control: composición por género de los tratados

El género no entra al modelo; esto controla que la política no lo discrimine de rebote a través de otras variables.

| genero | poblacion | tratados_mochila | razon |
|---|---|---|---|
| U | 0.468 | 0.483 | 1.031 |
| F | 0.367 | 0.347 | 0.947 |
| M | 0.165 | 0.170 | 1.028 |
