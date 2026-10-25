-- =============================================================================
-- 01_tipos.sql — de staging (texto) a tablas con tipos
--
--   stg_clients    -> clientes
--   stg_train      -> experimento
--   stg_purchases  -> compras        (una fila por PRODUCTO comprado)
--                  -> transacciones  (una fila por TICKET)
--
-- También guarda en control_conversiones cuántos valores no se pudieron
-- convertir en cada columna, y después borra stg_purchases (ocupa varios GB).
-- Se ejecuta con: python -m src.construir
-- =============================================================================


-- ── Conversiones seguras ────────────────────────────────────────────────────
-- Un cast inválido (por ejemplo 'abc'::numeric) aborta toda la consulta y
-- Postgres no tiene TRY_CAST. Estas funciones validan el formato primero y
-- devuelven NULL si no es válido.

CREATE OR REPLACE FUNCTION num_o_nulo(t text) RETURNS numeric
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
    SELECT CASE
        WHEN t ~ '^\s*[-+]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][-+]?[0-9]+)?\s*$'
        THEN t::numeric
    END
$$;

CREATE OR REPLACE FUNCTION ts_o_nulo(t text) RETURNS timestamp
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
    SELECT CASE
        WHEN t ~ '^\s*\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?\s*$'
        THEN t::timestamp
    END
$$;


-- ── Control de conversiones (antes de borrar staging) ───────────────────────
DROP TABLE IF EXISTS control_conversiones;

CREATE TABLE control_conversiones AS
WITH valores AS (
    SELECT 'clients' AS tabla, v.columna, v.texto, v.tipo
    FROM stg_clients s
    CROSS JOIN LATERAL (VALUES
        ('first_issue_date', s.first_issue_date, 'ts'),
        ('first_redeem_date', s.first_redeem_date, 'ts'),
        ('age', s.age, 'num')
    ) AS v(columna, texto, tipo)
    UNION ALL
    SELECT 'train', v.columna, v.texto, v.tipo
    FROM stg_train s
    CROSS JOIN LATERAL (VALUES
        ('treatment_flg', s.treatment_flg, 'num'),
        ('target', s.target, 'num')
    ) AS v(columna, texto, tipo)
    UNION ALL
    SELECT 'purchases', v.columna, v.texto, v.tipo
    FROM stg_purchases s
    CROSS JOIN LATERAL (VALUES
        ('transaction_datetime', s.transaction_datetime, 'ts'),
        ('purchase_sum', s.purchase_sum, 'num'),
        ('product_quantity', s.product_quantity, 'num'),
        ('regular_points_received', s.regular_points_received, 'num'),
        ('express_points_received', s.express_points_received, 'num'),
        ('regular_points_spent', s.regular_points_spent, 'num'),
        ('express_points_spent', s.express_points_spent, 'num')
    ) AS v(columna, texto, tipo)
)
SELECT
    tabla,
    columna,
    count(*) FILTER (WHERE texto IS NOT NULL) AS con_valor,
    count(*) FILTER (
        WHERE texto IS NOT NULL
          AND CASE tipo WHEN 'num' THEN num_o_nulo(texto) IS NULL
                        ELSE ts_o_nulo(texto) IS NULL END
    ) AS no_convertidos
FROM valores
GROUP BY tabla, columna;


-- ── clientes ────────────────────────────────────────────────────────────────
DROP TABLE IF EXISTS clientes CASCADE;

CREATE TABLE clientes AS
SELECT
    trim(client_id)                       AS client_id,
    ts_o_nulo(first_issue_date)           AS fecha_alta,
    ts_o_nulo(first_redeem_date)          AS fecha_primer_canje,
    num_o_nulo(age)::integer              AS edad_declarada,
    nullif(trim(gender), '')              AS genero
FROM stg_clients
WHERE nullif(trim(client_id), '') IS NOT NULL;

CREATE INDEX ON clientes (client_id);


-- ── experimento ─────────────────────────────────────────────────────────────
DROP TABLE IF EXISTS experimento CASCADE;

CREATE TABLE experimento AS
SELECT
    trim(client_id)                       AS client_id,
    num_o_nulo(treatment_flg)::integer    AS tratamiento,
    num_o_nulo(target)::integer           AS compro
FROM stg_train
WHERE nullif(trim(client_id), '') IS NOT NULL;

CREATE INDEX ON experimento (client_id);


-- ── compras (una fila por producto) ─────────────────────────────────────────
DROP TABLE IF EXISTS compras CASCADE;

CREATE TABLE compras AS
SELECT
    trim(client_id)                              AS client_id,
    trim(transaction_id)                         AS transaction_id,
    ts_o_nulo(transaction_datetime)              AS fecha,
    num_o_nulo(purchase_sum)                     AS monto_ticket,   -- total del ticket, repetido en cada línea
    trim(store_id)                               AS store_id,
    trim(product_id)                             AS product_id,
    num_o_nulo(product_quantity)                 AS cantidad,
    coalesce(num_o_nulo(regular_points_received), 0)
      + coalesce(num_o_nulo(express_points_received), 0)        AS puntos_ganados,
    -- los puntos gastados vienen con signo negativo en algunas versiones
    abs(coalesce(num_o_nulo(regular_points_spent), 0))
      + abs(coalesce(num_o_nulo(express_points_spent), 0))      AS puntos_canjeados
FROM stg_purchases
WHERE nullif(trim(client_id), '') IS NOT NULL;

CREATE INDEX ON compras (client_id);
CREATE INDEX ON compras (transaction_id);


-- ── transacciones (una fila por ticket) ─────────────────────────────────────
-- Error clásico a evitar: promediar el monto sobre líneas de producto. Un
-- ticket con 20 productos pesaría 20 veces más. Primero se agrega por ticket.
DROP TABLE IF EXISTS transacciones CASCADE;

CREATE TABLE transacciones AS
SELECT
    client_id,
    transaction_id,
    min(fecha)                                   AS fecha,
    max(monto_ticket)                            AS monto,
    min(monto_ticket) <> max(monto_ticket)       AS monto_inconsistente,
    max(puntos_ganados)                          AS puntos_ganados,
    max(puntos_canjeados)                        AS puntos_canjeados,
    min(store_id)                                AS store_id,
    count(*)                                     AS n_lineas,
    sum(cantidad)                                AS unidades
FROM compras
GROUP BY client_id, transaction_id;

CREATE INDEX ON transacciones (client_id, fecha);


-- ── Limpieza ────────────────────────────────────────────────────────────────
DROP TABLE IF EXISTS stg_purchases;

ANALYZE clientes;
ANALYZE experimento;
ANALYZE compras;
ANALYZE transacciones;
