# Etapa 3 — Simulación con efecto conocido

5 réplicas por escenario; cada método se entrena con el 70% y se evalúa en el 30%.
Las variables son las reales de 400,162 clientes; el resultado se genera con un efecto individual conocido.

## Escenarios (se varía un factor por vez respecto del base)

| escenario | clientes | efecto promedio | heterogeneidad | no molestar |
|---|---|---|---|---|
| base | 20,000 | 0.03 | 1.0 | 10% |
| efecto chico | 20,000 | 0.01 | 1.0 | 10% |
| sin heterogeneidad | 20,000 | 0.03 | 0.0 | 0% |
| mucha heterogeneidad | 20,000 | 0.03 | 2.5 | 10% |
| sin no molestar | 20,000 | 0.03 | 1.0 | 0% |
| muestra chica | 5,000 | 0.03 | 1.0 | 10% |
| muestra grande | 80,000 | 0.03 | 1.0 | 10% |

## Escenario base

| metodo | pehe | captura | qini_relativo | sesgo_ate | segundos |
|---|---|---|---|---|---|
| Sin modelo (efecto promedio) | 0.0435 | 0.4404 | -0.3140 | -0.0015 | 0.0000 |
| S-learner | 0.0381 | 0.8617 | 0.6465 | -0.0137 | 0.1291 |
| T-learner | 0.0754 | 0.7560 | 0.4859 | -0.0026 | 0.1669 |
| X-learner | 0.0401 | 0.7819 | 0.5982 | -0.0022 | 0.2875 |
| Variable transformada | 0.0830 | 0.6019 | 0.1795 | -0.0004 | 0.0447 |
| DR-learner | 0.0496 | 0.7029 | 0.4897 | -0.0014 | 0.8974 |

- **PEHE**: error en el efecto de cada cliente. *Sin modelo (efecto promedio)* asigna a todos el mismo efecto: su PEHE es el desvío del efecto real.
- **Captura**: tratando al 30% con mayor uplift predicho, qué fracción se obtiene de lo que ganaría la mejor elección posible. Sin modelo, la elección es al azar.
- **Qini relativo**: coeficiente Qini (calculado con compras observadas, como en datos reales) sobre el del oráculo que ordena por el efecto real. Es mucho más ruidoso que el PEHE y la captura, que usan el efecto real: con pocos clientes de prueba, hasta el ordenamiento al azar puede salir bien. Es la única de las tres métricas disponible en la etapa 4.

## Mejor método por escenario

| escenario | mejor en PEHE | PEHE | PEHE sin modelo | mejor en captura | captura | captura sin modelo |
|---|---|---|---|---|---|---|
| base | S-learner | 0.0381 | 0.0435 | S-learner | 0.8617 | 0.4404 |
| efecto chico | S-learner | 0.0300 | 0.0325 | S-learner | 0.7312 | 0.3024 |
| sin heterogeneidad | S-learner | 0.0255 | 0.0133 | S-learner | 0.8565 | 0.7413 |
| mucha heterogeneidad | S-learner | 0.0411 | 0.0488 | S-learner | 0.7967 | 0.3565 |
| sin no molestar | S-learner | 0.0242 | 0.0199 | S-learner | 0.7788 | 0.5796 |
| muestra chica | S-learner | 0.0466 | 0.0436 | S-learner | 0.6495 | 0.4276 |
| muestra grande | X-learner | 0.0239 | 0.0422 | S-learner | 0.9083 | 0.4376 |

![PEHE por escenario](pehe.png)

![Captura por escenario](captura.png)

![Curvas Qini](qini_base.png)
