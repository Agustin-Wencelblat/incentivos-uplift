# Etapa 4 — Uplift en el experimento real

- Clientes: 200,039; entrenamiento 139,582, prueba 60,457
- Efecto promedio en prueba: 0.0266 [0.0189, 0.0344]
- Modelo elegido por Qini en validación cruzada (3 pliegues, solo entrenamiento): **X-learner**

## Coeficiente Qini en prueba (IC 95% bootstrap, 500 remuestras pareadas)

| metodo | qini_cv | qini | ic_inf | ic_sup | prob_qini_positivo | dif_vs_elegido_ic_inf | dif_vs_elegido_ic_sup |
|---|---|---|---|---|---|---|---|
| S-learner | 0.00612 | 0.00277 | 0.00051 | 0.00513 | 0.99400 | -0.00241 | 0.00072 |
| T-learner | 0.00516 | 0.00260 | 0.00044 | 0.00498 | 0.99200 | -0.00266 | 0.00083 |
| X-learner | 0.00664 | 0.00350 | 0.00137 | 0.00592 | 1.00000 | 0.00000 | 0.00000 |
| Variable transformada | 0.00399 | 0.00299 | 0.00094 | 0.00522 | 0.99600 | -0.00291 | 0.00187 |
| DR-learner | 0.00632 | 0.00229 | 0.00013 | 0.00470 | 0.97800 | -0.00291 | 0.00048 |

- `qini_cv`: media en validación cruzada (la que decide la elección). `qini`: en prueba.
- `dif_vs_elegido`: intervalo de la diferencia con el modelo elegido sobre las mismas remuestras. Si contiene al 0, la prueba no distingue a ese modelo del elegido.
- El modelo elegido ordena mejor que el azar con probabilidad bootstrap 1.00.

![Curvas Qini](curvas_qini.png)

## Calibración por decil (X-learner)

- Decil de mayor uplift predicho: predicho 0.1082, observado 0.0818 [0.0567, 0.1069]
- Decil de menor uplift predicho: predicho -0.0038, observado 0.0296 [0.0048, 0.0544]
- Pendiente de observado sobre predicho: 0.56 (1 = calibrado; menor que 1 = el modelo exagera las diferencias entre clientes)

| grupo | clientes | uplift_predicho | efecto_observado | ic_inf | ic_sup |
|---|---|---|---|---|---|
| 1 | 6045 | 0.1082 | 0.0818 | 0.0567 | 0.1069 |
| 2 | 6046 | 0.0615 | 0.0316 | 0.0064 | 0.0567 |
| 3 | 6046 | 0.0477 | 0.0332 | 0.0081 | 0.0582 |
| 4 | 6045 | 0.0386 | 0.0329 | 0.0081 | 0.0576 |
| 5 | 6046 | 0.0318 | -0.0031 | -0.0275 | 0.0213 |
| 6 | 6046 | 0.0262 | 0.0094 | -0.0143 | 0.0332 |
| 7 | 6045 | 0.0212 | 0.0049 | -0.0179 | 0.0278 |
| 8 | 6046 | 0.0164 | 0.0137 | -0.0090 | 0.0365 |
| 9 | 6046 | 0.0107 | 0.0244 | 0.0021 | 0.0467 |
| 10 | 6046 | -0.0038 | 0.0296 | 0.0048 | 0.0544 |

![Calibración](calibracion.png)
