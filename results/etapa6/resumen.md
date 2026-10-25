# Etapa 6 — Validación y monitoreo de drift

Referencia: 200,760 clientes; datos "actuales": la otra mitad (199,402), con cambios conocidos.
Umbrales de PSI: > 0.1 revisar, > 0.25 alerta.

## Escenarios

| escenario | errores_validacion | avisos_validacion | detalle_validacion | variables_en_alerta | variables_a_revisar | variable_mas_afectada | psi_max | sliced_wasserstein | psi_prediccion | cambio_tau_medio |
|---|---|---|---|---|---|---|---|---|---|---|
| sin cambios | 0 | 0 | - | 0 | 0 | ticket_mediano | 0.000 | 0.004 | 0.000 | 0.000 |
| inflación 30% | 0 | 0 | - | 0 | 2 | ticket_medio | 0.158 | 0.129 | 0.002 | -0.002 |
| clientes inactivos | 0 | 1 | dias_desde_ultima: 100.0% de valores fuera del rango de entrenamiento | 3 | 0 | dias_desde_ultima | 8.283 | 0.974 | 0.231 | -0.009 |
| más clientes nuevos | 0 | 0 | - | 0 | 6 | n_transacciones | 0.210 | 0.144 | 0.004 | 0.003 |
| falla de datos | 1 | 0 | edad: faltantes 40.3% (en entrenamiento 0.8%) | 1 | 0 | edad | 1.747 | 0.014 | 0.007 | -0.003 |

- `errores_validacion`: lo que haría frenar el pipeline antes de puntuar. La validación no distingue una falla de carga de un cambio real de la población (por ejemplo, más clientes nuevos implica más faltantes en las variables que necesitan varias compras): frena y deja la decisión a una persona.
- `psi_prediccion`: PSI de la distribución del uplift predicho. Un cambio en las variables que el modelo casi no usa puede no moverla; uno en las que usa, sí.

![PSI por variable y escenario](psi.png)

## Falsas alarmas: PSI entre dos muestras de la misma población

| clientes_por_muestra | psi_medio | psi_esperado | frac_revisar | frac_alerta |
|---|---|---|---|---|
| 200 | 0.0847 | 0.0900 | 0.3409 | 0.0045 |
| 500 | 0.0321 | 0.0360 | 0.0045 | 0.0000 |
| 2000 | 0.0085 | 0.0090 | 0.0000 | 0.0000 |
| 10000 | 0.0017 | 0.0018 | 0.0000 | 0.0000 |
| 50000 | 0.0003 | 0.0004 | 0.0000 | 0.0000 |

Con muestras chicas, el PSI supera los umbrales habituales sin que haya ningún cambio: el umbral tiene que tener en cuenta el tamaño de muestra (el PSI esperado sin drift es aproximadamente (bins − 1)(1/n_ref + 1/n_act)).
