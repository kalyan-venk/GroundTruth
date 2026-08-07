-- Covariate balance, as a single SQL statement.
--
-- The standardised mean difference per feature, which is the check that
-- catches what SRM cannot: arms with the right number of users but the wrong
-- kind. Written long-form with a UNION ALL per feature rather than unpivoted,
-- because the explicit version is what someone reviewing the logic wants to
-- read.
--
--   SMD = (mean_treatment - mean_control) / sqrt((var_t + var_c) / 2)
--
-- Convention: |SMD| < 0.1 is negligible regardless of sample size. That is the
-- whole reason to compute it rather than a t-test. At 14M rows a t-test
-- rejects on differences far too small to matter.

WITH arm_stats AS (
    SELECT
        treatment,
        COUNT(*) AS n,
        AVG(f0) AS m_f0, VAR_SAMP(f0) AS v_f0,
        AVG(f1) AS m_f1, VAR_SAMP(f1) AS v_f1,
        AVG(f2) AS m_f2, VAR_SAMP(f2) AS v_f2,
        AVG(f3) AS m_f3, VAR_SAMP(f3) AS v_f3,
        AVG(f4) AS m_f4, VAR_SAMP(f4) AS v_f4,
        AVG(f5) AS m_f5, VAR_SAMP(f5) AS v_f5
    FROM read_csv_auto(:source)
    GROUP BY treatment
),
t AS (SELECT * FROM arm_stats WHERE treatment = 1),
c AS (SELECT * FROM arm_stats WHERE treatment = 0)

-- The UNION has to be wrapped before ordering: ORDER BY on a set operation can
-- only reference the output columns directly, not an expression over them.
SELECT feature, difference, smd
FROM (
    SELECT 'f0' AS feature, t.m_f0 - c.m_f0 AS difference,
           (t.m_f0 - c.m_f0) / SQRT((t.v_f0 + c.v_f0) / 2) AS smd FROM t, c
    UNION ALL
    SELECT 'f1', t.m_f1 - c.m_f1,
           (t.m_f1 - c.m_f1) / SQRT((t.v_f1 + c.v_f1) / 2) FROM t, c
    UNION ALL
    SELECT 'f2', t.m_f2 - c.m_f2,
           (t.m_f2 - c.m_f2) / SQRT((t.v_f2 + c.v_f2) / 2) FROM t, c
    UNION ALL
    SELECT 'f3', t.m_f3 - c.m_f3,
           (t.m_f3 - c.m_f3) / SQRT((t.v_f3 + c.v_f3) / 2) FROM t, c
    UNION ALL
    SELECT 'f4', t.m_f4 - c.m_f4,
           (t.m_f4 - c.m_f4) / SQRT((t.v_f4 + c.v_f4) / 2) FROM t, c
    UNION ALL
    SELECT 'f5', t.m_f5 - c.m_f5,
           (t.m_f5 - c.m_f5) / SQRT((t.v_f5 + c.v_f5) / 2) FROM t, c
) AS smds
ORDER BY ABS(smd) DESC;
