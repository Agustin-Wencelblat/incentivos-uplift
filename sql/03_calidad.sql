-- =============================================================================
-- 03_calidad.sql — controles de calidad
-- Abrir en pgAdmin (Query Tool) y ejecutar cada consulta por separado
-- Requiere haber corrido: python -m src.construir
-- =============================================================================


-- 1. Valores que no se pudieron convertir (calculado al construir las tablas)
SELECT tabla, columna, con_valor, no_convertidos,
       round(100.0 * no_convertidos / nullif(con_valor, 0), 3) AS pct_no_convertidos
FROM control_conversiones
ORDER BY no_convertidos DESC, tabla, columna;


-- 2. Rango de fechas del historial. Las compras tienen que ser ANTERIORES a
--    la comunicación; si hubiera compras posteriores serían fuga de información.
SELECT min(fecha) AS primera_compra, max(fecha) AS ultima_compra,
       count(*) AS transacciones, count(DISTINCT client_id) AS clientes
FROM transacciones;


-- 3. ¿El monto del ticket es el mismo en todas las líneas de un ticket?
--    (El SQL asume que purchase_sum es el total del ticket, repetido por línea.)
SELECT count(*) FILTER (WHERE monto_inconsistente) AS tickets_inconsistentes,
       count(*)                                     AS tickets,
       round(100.0 * avg(monto_inconsistente::int), 3) AS pct
FROM transacciones;


-- 4. Duplicados y cobertura del experimento
SELECT
    (SELECT count(*) FROM experimento)                                   AS filas_experimento,
    (SELECT count(DISTINCT client_id) FROM experimento)                  AS clientes_distintos,
    (SELECT count(*) FROM experimento e
      WHERE NOT EXISTS (SELECT 1 FROM clientes c WHERE c.client_id = e.client_id))
                                                                         AS sin_ficha_de_cliente,
    (SELECT count(*) FROM experimento e
      WHERE NOT EXISTS (SELECT 1 FROM transacciones t WHERE t.client_id = e.client_id))
                                                                         AS sin_compras_previas;


-- 5. Valores de tratamiento y resultado (tienen que ser solo 0 y 1)
SELECT tratamiento, compro, count(*) AS clientes
FROM experimento
GROUP BY tratamiento, compro
ORDER BY tratamiento, compro;


-- 6. Valores sospechosos
SELECT
    count(*) FILTER (WHERE monto < 0)               AS tickets_monto_negativo,
    count(*) FILTER (WHERE monto = 0)               AS tickets_monto_cero,
    count(*) FILTER (WHERE unidades <= 0)           AS tickets_sin_unidades,
    percentile_cont(0.999) WITHIN GROUP (ORDER BY monto) AS monto_p999,
    max(monto)                                      AS monto_maximo
FROM transacciones;

SELECT
    count(*) FILTER (WHERE edad_declarada < 16 OR edad_declarada > 100) AS edades_fuera_de_rango,
    count(*) FILTER (WHERE edad_declarada IS NULL)                       AS edades_nulas,
    count(*) FILTER (WHERE fecha_primer_canje < fecha_alta)              AS canje_antes_del_alta,
    count(*)                                                             AS clientes
FROM clientes;


-- 6b. FUGA DE INFORMACIÓN: fechas de clients posteriores al corte (la última
--     compra registrada, previa a la comunicación). 02_variables.sql ya las
--     ignora; esta consulta muestra cuántas son y si se relacionan con el
--     resultado del experimento. Si los clientes con primer canje posterior al
--     corte compran mucho más durante la campaña, usar esa fecha habría sido
--     una fuga grave.
SELECT
    (SELECT fecha_corte FROM parametros)                                AS fecha_corte,
    count(*) FILTER (WHERE c.fecha_primer_canje > p.fecha_corte)        AS canjes_posteriores_al_corte,
    count(*) FILTER (WHERE c.fecha_alta > p.fecha_corte)                AS altas_posteriores_al_corte,
    max(c.fecha_primer_canje)                                           AS ultimo_primer_canje
FROM clientes c CROSS JOIN parametros p;

SELECT
    CASE
        WHEN c.fecha_primer_canje IS NULL            THEN '1. nunca canjeó'
        WHEN c.fecha_primer_canje <= p.fecha_corte   THEN '2. canjeó antes del corte'
        ELSE                                              '3. primer canje DESPUÉS del corte'
    END                                    AS grupo,
    count(*)                               AS clientes,
    round(avg(e.compro), 4)                AS tasa_compra_en_la_campania
FROM experimento e
JOIN clientes c USING (client_id)
CROSS JOIN parametros p
GROUP BY 1
ORDER BY 1;


-- 7. Porcentaje de nulos por variable (clientes del experimento)
SELECT
    round(100.0 * avg((ticket_medio IS NULL)::int), 2)           AS pct_nulo_ticket,
    round(100.0 * avg((intervalo_medio_dias IS NULL)::int), 2)   AS pct_nulo_intervalo,
    round(100.0 * avg((tendencia_gasto IS NULL)::int), 2)        AS pct_nulo_tendencia,
    round(100.0 * avg((edad IS NULL)::int), 2)                   AS pct_nulo_edad,
    round(100.0 * avg((antiguedad_dias IS NULL)::int), 2)        AS pct_nulo_antiguedad,
    round(100.0 * avg((dias_hasta_primer_canje IS NULL)::int), 2) AS pct_nulo_primer_canje
FROM datos_experimento;
