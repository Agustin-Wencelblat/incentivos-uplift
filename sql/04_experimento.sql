-- =============================================================================
-- 04_experimento.sql — primera mirada al experimento (pgAdmin)
-- El análisis completo, con intervalos y CUPED, está en
-- python -m src.estudios.experimento
-- =============================================================================


-- 1. Proporción de tratados y tasa de compra por grupo
SELECT
    tratamiento,
    count(*)                                              AS clientes,
    round(100.0 * count(*) / sum(count(*)) OVER (), 2)    AS pct_clientes,
    round(avg(compro), 4)                                 AS tasa_compra
FROM datos_experimento
GROUP BY tratamiento
ORDER BY tratamiento;


-- 2. Efecto promedio con intervalo del 95% (aproximación normal)
WITH g AS (
    SELECT
        avg(compro) FILTER (WHERE tratamiento = 1)        AS p1,
        avg(compro) FILTER (WHERE tratamiento = 0)        AS p0,
        count(*)    FILTER (WHERE tratamiento = 1)        AS n1,
        count(*)    FILTER (WHERE tratamiento = 0)        AS n0
    FROM datos_experimento
)
SELECT
    round(p1 - p0, 5)                                                        AS efecto,
    round(sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0), 5)                  AS error_estandar,
    round(p1 - p0 - 1.96 * sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0), 5) AS ic_inferior,
    round(p1 - p0 + 1.96 * sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0), 5) AS ic_superior
FROM g;


-- 3. Balance: diferencia estandarizada de medias (SMD) entre tratados y control.
--    Con asignación al azar, |SMD| < 0.1 en todas las variables.
WITH largo AS (
    SELECT d.tratamiento, v.variable, v.valor
    FROM datos_experimento d
    CROSS JOIN LATERAL (VALUES
        ('n_transacciones', d.n_transacciones::float8),
        ('gasto_total', d.gasto_total::float8),
        ('ticket_medio', d.ticket_medio::float8),
        ('dias_desde_ultima', d.dias_desde_ultima::float8),
        ('n_trans_30d', d.n_trans_30d::float8),
        ('puntos_canjeados', d.puntos_canjeados::float8),
        ('edad', d.edad::float8),
        ('antiguedad_dias', d.antiguedad_dias::float8),
        ('canjeo_alguna_vez', d.canjeo_alguna_vez::float8)
    ) AS v(variable, valor)
),
por_grupo AS (
    SELECT
        variable,
        avg(valor) FILTER (WHERE tratamiento = 1)          AS media_t,
        avg(valor) FILTER (WHERE tratamiento = 0)          AS media_c,
        var_samp(valor) FILTER (WHERE tratamiento = 1)     AS var_t,
        var_samp(valor) FILTER (WHERE tratamiento = 0)     AS var_c
    FROM largo
    GROUP BY variable
)
SELECT
    variable,
    round(media_t::numeric, 3) AS media_tratados,
    round(media_c::numeric, 3) AS media_control,
    round(((media_t - media_c) / nullif(sqrt((var_t + var_c) / 2), 0))::numeric, 4) AS smd
FROM por_grupo
ORDER BY abs((media_t - media_c) / nullif(sqrt((var_t + var_c) / 2), 0)) DESC;


-- 4. Efecto por segmento de frecuencia de compra (primera señal de heterogeneidad)
WITH seg AS (
    SELECT *, ntile(4) OVER (ORDER BY n_transacciones) AS cuartil_frecuencia
    FROM datos_experimento
)
SELECT
    cuartil_frecuencia,
    min(n_transacciones)                                       AS desde,
    max(n_transacciones)                                       AS hasta,
    count(*)                                                   AS clientes,
    round(avg(compro) FILTER (WHERE tratamiento = 1), 4)       AS tasa_tratados,
    round(avg(compro) FILTER (WHERE tratamiento = 0), 4)       AS tasa_control,
    round(avg(compro) FILTER (WHERE tratamiento = 1)
        - avg(compro) FILTER (WHERE tratamiento = 0), 4)       AS efecto
FROM seg
GROUP BY cuartil_frecuencia
ORDER BY cuartil_frecuencia;
