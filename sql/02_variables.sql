-- =============================================================================
-- 02_variables.sql — una fila por cliente con sus variables
--
-- Todas las variables usan solo compras ANTERIORES a la fecha de corte
-- (corrección en el tiempo: el modelo no puede ver información posterior a la
-- decisión). La fecha de corte es la última compra registrada, porque el
-- historial de X5 termina antes de la comunicación.
-- Se ejecuta con: python -m src.construir
-- =============================================================================

DROP TABLE IF EXISTS parametros;
CREATE TABLE parametros AS
SELECT max(fecha) AS fecha_corte FROM transacciones;


DROP TABLE IF EXISTS variables_clientes CASCADE;

CREATE TABLE variables_clientes AS
WITH
corte AS (
    SELECT fecha_corte FROM parametros
),

trans AS (
    SELECT t.*
    FROM transacciones t, corte
    WHERE t.fecha <= corte.fecha_corte
),

-- Días entre compras consecutivas del mismo cliente
intervalos AS (
    SELECT
        client_id,
        EXTRACT(EPOCH FROM fecha - LAG(fecha) OVER (PARTITION BY client_id ORDER BY fecha)) / 86400.0
            AS dias_entre_compras
    FROM trans
),

resumen_intervalos AS (
    SELECT
        client_id,
        avg(dias_entre_compras)         AS intervalo_medio_dias,
        stddev_samp(dias_entre_compras) AS intervalo_desvio_dias
    FROM intervalos
    WHERE dias_entre_compras IS NOT NULL
    GROUP BY client_id
),

resumen_trans AS (
    SELECT
        t.client_id,
        count(*)                                                     AS n_transacciones,
        sum(t.monto)                                                 AS gasto_total,
        avg(t.monto)                                                 AS ticket_medio,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY t.monto)         AS ticket_mediano,
        EXTRACT(EPOCH FROM c.fecha_corte - max(t.fecha)) / 86400.0   AS dias_desde_ultima,
        EXTRACT(EPOCH FROM c.fecha_corte - min(t.fecha)) / 86400.0   AS dias_desde_primera,
        count(*) FILTER (WHERE t.fecha > c.fecha_corte - INTERVAL '30 days')    AS n_trans_30d,
        coalesce(sum(t.monto) FILTER (WHERE t.fecha > c.fecha_corte - INTERVAL '30 days'), 0)
                                                                     AS gasto_30d,
        coalesce(sum(t.monto) FILTER (WHERE t.fecha <= c.fecha_corte - INTERVAL '30 days'
                                        AND t.fecha >  c.fecha_corte - INTERVAL '120 days'), 0)
                                                                     AS gasto_31_120d,
        sum(t.puntos_ganados)                                        AS puntos_ganados,
        sum(t.puntos_canjeados)                                      AS puntos_canjeados,
        avg((t.puntos_canjeados > 0)::int)                           AS frac_trans_con_canje,
        count(DISTINCT t.store_id)                                   AS n_tiendas,
        avg(t.unidades)                                              AS unidades_por_transaccion
    FROM trans t, corte c
    GROUP BY t.client_id, c.fecha_corte
),

-- La única variable que necesita el detalle por producto
resumen_productos AS (
    SELECT co.client_id, count(DISTINCT co.product_id) AS n_productos
    FROM compras co, corte c
    WHERE co.fecha <= c.fecha_corte
    GROUP BY co.client_id
)

SELECT
    cl.client_id,
    cl.genero,                         -- NO entra al modelo; solo para control

    coalesce(rt.n_transacciones, 0)    AS n_transacciones,
    coalesce(rt.gasto_total, 0)        AS gasto_total,
    rt.ticket_medio,
    rt.ticket_mediano,
    rt.dias_desde_ultima,
    rt.dias_desde_primera,
    coalesce(rt.n_trans_30d, 0)        AS n_trans_30d,
    coalesce(rt.gasto_30d, 0)          AS gasto_30d,
    coalesce(rt.gasto_31_120d, 0)      AS gasto_31_120d,
    -- Gasto de los últimos 30 días contra el promedio mensual de los 90 previos.
    -- > 1: el cliente está comprando más que antes.
    rt.gasto_30d / nullif(rt.gasto_31_120d / 3.0, 0)  AS tendencia_gasto,
    ri.intervalo_medio_dias,
    ri.intervalo_desvio_dias,
    coalesce(rt.puntos_ganados, 0)     AS puntos_ganados,
    coalesce(rt.puntos_canjeados, 0)   AS puntos_canjeados,
    rt.frac_trans_con_canje,
    coalesce(rt.n_tiendas, 0)          AS n_tiendas,
    coalesce(rp.n_productos, 0)        AS n_productos,
    rt.unidades_por_transaccion,

    -- Edad declarada: valores fuera de un rango plausible se tratan como faltantes
    CASE WHEN cl.edad_declarada BETWEEN 16 AND 100 THEN cl.edad_declarada END   AS edad,

    -- Antigüedad: solo si el alta es anterior al corte (si no, al momento de la
    -- decisión el cliente todavía no existía para el negocio)
    CASE WHEN cl.fecha_alta <= c.fecha_corte
         THEN EXTRACT(EPOCH FROM c.fecha_corte - cl.fecha_alta) / 86400.0 END    AS antiguedad_dias,

    -- FUGA DE INFORMACIÓN EVITADA: first_redeem_date puede ser POSTERIOR a la
    -- comunicación. Un primer canje después de la campaña implica una compra
    -- durante la campaña, que es justo lo que se quiere predecir. Por eso el
    -- canje solo cuenta si ocurrió antes del corte; si fue después, para el
    -- modelo ese cliente "todavía no había canjeado".
    -- (En los notebooks de ejemplo de scikit-uplift, la fecha de primer canje
    -- sin este filtro es la variable más importante del modelo: es una señal
    -- de alarma típica de fuga.)
    CASE WHEN cl.fecha_primer_canje <= c.fecha_corte
         THEN EXTRACT(EPOCH FROM cl.fecha_primer_canje - cl.fecha_alta) / 86400.0
    END                                                                          AS dias_hasta_primer_canje,
    (cl.fecha_primer_canje IS NOT NULL
     AND cl.fecha_primer_canje <= c.fecha_corte)::int                            AS canjeo_alguna_vez

FROM clientes cl
CROSS JOIN corte c
LEFT JOIN resumen_trans      rt ON rt.client_id = cl.client_id
LEFT JOIN resumen_intervalos ri ON ri.client_id = cl.client_id
LEFT JOIN resumen_productos  rp ON rp.client_id = cl.client_id;

CREATE INDEX ON variables_clientes (client_id);
ANALYZE variables_clientes;


-- Vista de modelado: clientes del experimento con tratamiento y resultado
DROP VIEW IF EXISTS datos_experimento;
CREATE VIEW datos_experimento AS
SELECT v.*, e.tratamiento, e.compro
FROM experimento e
JOIN variables_clientes v USING (client_id);
