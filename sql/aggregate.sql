-- The stage-1 aggregation, as SQL.
--
-- The Python in src/ingest.py builds this with the DataFrame API because it
-- generates 100+ aggregation expressions programmatically. That is the right
-- tool for a wide, repetitive aggregation. But the logic underneath is a
-- perfectly ordinary GROUP BY, and writing it out makes that visible -- both
-- to a reader and to tests/test_sql.py, which runs this query in DuckDB and
-- asserts it matches the Spark aggregate to 10 significant figures.
--
-- Two independent implementations agreeing is a stronger correctness argument
-- than either one alone.
--
-- Parameterised by :source, which is either the raw CSV or a sample of it.

SELECT
    treatment,

    -- Counts and the primary metric.
    COUNT(*)                                     AS n,
    SUM(conversion)                              AS sum_y,
    SUM(conversion * conversion)                 AS sum_yy,

    -- The naive covariate. Measured during the experiment, kept only as a
    -- deliberate contrast -- see the CUPED stage.
    SUM(visit)                                   AS sum_xn,
    SUM(visit * visit)                           AS sum_xnxn,
    SUM(conversion * visit)                      AS sum_yxn,

    -- Compliance. Exposure is post-randomisation, so it never enters the
    -- effect estimate; it is here for the CACE denominator only.
    SUM(exposure)                                AS sum_exposure,

    -- Per-feature first moments and the feature-outcome cross-products that a
    -- full 12-covariate ANCOVA needs.
    SUM(f0)  AS sum_f0,   SUM(f0  * conversion) AS sum_f0_y,
    SUM(f1)  AS sum_f1,   SUM(f1  * conversion) AS sum_f1_y,
    SUM(f2)  AS sum_f2,   SUM(f2  * conversion) AS sum_f2_y,
    SUM(f3)  AS sum_f3,   SUM(f3  * conversion) AS sum_f3_y,
    SUM(f4)  AS sum_f4,   SUM(f4  * conversion) AS sum_f4_y,
    SUM(f5)  AS sum_f5,   SUM(f5  * conversion) AS sum_f5_y,
    SUM(f6)  AS sum_f6,   SUM(f6  * conversion) AS sum_f6_y,
    SUM(f7)  AS sum_f7,   SUM(f7  * conversion) AS sum_f7_y,
    SUM(f8)  AS sum_f8,   SUM(f8  * conversion) AS sum_f8_y,
    SUM(f9)  AS sum_f9,   SUM(f9  * conversion) AS sum_f9_y,
    SUM(f10) AS sum_f10,  SUM(f10 * conversion) AS sum_f10_y,
    SUM(f11) AS sum_f11,  SUM(f11 * conversion) AS sum_f11_y

FROM read_csv_auto(:source)
GROUP BY treatment
ORDER BY treatment;
