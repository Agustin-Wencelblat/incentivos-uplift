# Etapa 2 — El experimento

## ¿Fue al azar?

- Clientes: 39,854 (49.7% tratados)
- Mayor |SMD|: 0.0207 (n_transacciones); variables con |SMD| > 0.1: 0 de 22
- Ruido esperado del SMD por azar con este tamaño de muestra: ≈ 0.0100, así que el umbral usado es max(0.1, 3 × ruido) = 0.1000
- Variables por encima del umbral: 0
- AUC de un clasificador que intenta predecir el tratamiento: 0.500 (≈ 0.5 si fue al azar)

Sin señales de desbalance: compatible con una asignación al azar.

## ¿Hubo efecto?

- Tasa de compra: 0.6369 en tratados, 0.5994 en control
- Efecto promedio: 0.0376, IC 95% [0.0280, 0.0471], p = 1.1e-14
- IC bootstrap: [0.0280, 0.0476]
- Con CUPED: 0.0339, IC 95% [0.0250, 0.0428]
  - Correlación entre el resultado y n_transacciones: ρ = 0.361
  - Reducción de varianza esperada: ρ² = 13.0%; intervalo 6.7% más angosto

## Efecto por segmento (p-valores corregidos usando Holm)

| dimension | segmento | clientes | efecto | ic_inf | ic_sup | p_holm |
|---|---|---|---|---|---|---|
| Frecuencia | Q1 (menos frecuentes) | 9964 | 0.0406 | 0.0221 | 0.0591 | 0.0001 |
| Frecuencia | Q2 | 9963 | 0.0382 | 0.0186 | 0.0577 | 0.0005 |
| Frecuencia | Q3 | 9963 | 0.0348 | 0.0171 | 0.0525 | 0.0005 |
| Frecuencia | Q4 (más frecuentes) | 9964 | 0.0213 | 0.0085 | 0.0340 | 0.0011 |
| Usa el programa de puntos | no | 15401 | 0.0308 | 0.0150 | 0.0465 | 0.0005 |
| Usa el programa de puntos | sí | 24453 | 0.0395 | 0.0279 | 0.0511 | 0.0000 |
| Compró en los últimos 30 días | sí | 39854 | 0.0376 | 0.0280 | 0.0471 | 0.0000 |

![Efecto por segmento](segmentos.png)
