/*
FinGuard Pro — SQL Analytics
Database: finguard
Table: transactions
Dataset: PaySim synthetic mobile-money transactions

Purpose:
- Validate the imported dataset
- Calculate transaction and fraud KPIs
- Analyse fraud by transaction type and transaction amount
- Investigate sender/receiver balance patterns
- Practise CTEs and window functions

Important:
- isFraud is the dataset's label, not a prediction from our own model.
- Findings describe this synthetic dataset and should not be treated as real-world fraud rules.
- Run queries individually in MySQL Workbench when working with the full 6.36M-row table.
*/

USE finguard;

-- 01. DATASET VALIDATION
SELECT
    COUNT(*) AS total_rows,
    SUM(isFraud) AS fraud_count,
    SUM(isFlaggedFraud) AS flagged_count,
    MIN(step) AS first_step,
    MAX(step) AS last_step
FROM transactions;

-- Expected: 6362620 rows, 8213 fraud labels, 16 flagged rows, step 1 to 743.

-- 02. OVERALL TRANSACTION KPIs
SELECT
    COUNT(*) AS total_transactions,
    ROUND(SUM(amount), 2) AS total_transaction_amount,
    ROUND(AVG(amount), 2) AS average_transaction_amount,
    SUM(isFraud) AS total_fraud_transactions,
    ROUND(100.0 * AVG(isFraud), 4) AS fraud_percentage
FROM transactions;

-- 03. TRANSACTION VOLUME AND FRAUD BY TYPE
SELECT
    type,
    COUNT(*) AS total_transactions,
    SUM(isFraud) AS fraud_transactions,
    ROUND(100.0 * AVG(isFraud), 4) AS fraud_rate_percent,
    ROUND(SUM(amount), 2) AS total_amount,
    ROUND(AVG(amount), 2) AS average_amount
FROM transactions
GROUP BY type
ORDER BY fraud_rate_percent DESC;

-- 04. HIGH-VALUE TRANSACTIONS (amount >= 1,000,000)
-- Exploratory threshold only; not an official fraud rule.
SELECT
    type,
    COUNT(*) AS high_value_transactions,
    SUM(isFraud) AS fraud_count,
    ROUND(100.0 * AVG(isFraud), 2) AS fraud_rate_percent,
    ROUND(AVG(amount), 2) AS average_amount,
    ROUND(MAX(amount), 2) AS maximum_amount
FROM transactions
WHERE amount >= 1000000
GROUP BY type
ORDER BY fraud_rate_percent DESC;

-- 05. SENDER BALANCE MATCH RATE BY FRAUD LABEL
-- Match means sender balance decrease is within 0.01 of transaction amount.
SELECT
    isFraud,
    COUNT(*) AS total_transactions,
    SUM(
        CASE
            WHEN ABS((oldbalanceOrg - newbalanceOrig) - amount) <= 0.01
            THEN 1 ELSE 0
        END
    ) AS sender_balance_matches,
    ROUND(
        100.0 * AVG(
            CASE
                WHEN ABS((oldbalanceOrg - newbalanceOrig) - amount) <= 0.01
                THEN 1.0 ELSE 0.0
            END
        ), 2
    ) AS sender_match_rate_percent
FROM transactions
GROUP BY isFraud;

-- 06. RECEIVER BALANCE MATCH RATE BY FRAUD LABEL
-- Match means receiver balance increase is within 0.01 of transaction amount.
SELECT
    isFraud,
    COUNT(*) AS total_transactions,
    SUM(
        CASE
            WHEN ABS((newbalanceDest - oldbalanceDest) - amount) <= 0.01
            THEN 1 ELSE 0
        END
    ) AS receiver_balance_matches,
    ROUND(
        100.0 * AVG(
            CASE
                WHEN ABS((newbalanceDest - oldbalanceDest) - amount) <= 0.01
                THEN 1.0 ELSE 0.0
            END
        ), 2
    ) AS receiver_match_rate_percent
FROM transactions
GROUP BY isFraud;

-- 07. BALANCE MATCH RATES BY TYPE AND FRAUD LABEL
SELECT
    type,
    isFraud,
    COUNT(*) AS total_transactions,
    ROUND(
        100.0 * AVG(
            CASE
                WHEN ABS((oldbalanceOrg - newbalanceOrig) - amount) <= 0.01
                THEN 1.0 ELSE 0.0
            END
        ), 2
    ) AS sender_match_rate_percent,
    ROUND(
        100.0 * AVG(
            CASE
                WHEN ABS((newbalanceDest - oldbalanceDest) - amount) <= 0.01
                THEN 1.0 ELSE 0.0
            END
        ), 2
    ) AS receiver_match_rate_percent
FROM transactions
GROUP BY type, isFraud
ORDER BY type, isFraud;

-- 08. POSITIVE-AMOUNT TRANSACTIONS WITH ZERO SENDER BALANCES BY TYPE
-- This is a balance pattern, not a fraud verdict.
SELECT
    type,
    COUNT(*) AS total_transactions,
    SUM(
        CASE
            WHEN amount > 0 AND oldbalanceOrg = 0 AND newbalanceOrig = 0
            THEN 1 ELSE 0
        END
    ) AS zero_balance_transactions,
    ROUND(
        100.0 * AVG(
            CASE
                WHEN amount > 0 AND oldbalanceOrg = 0 AND newbalanceOrig = 0
                THEN 1.0 ELSE 0.0
            END
        ), 2
    ) AS percentage
FROM transactions
GROUP BY type
ORDER BY zero_balance_transactions DESC;

-- Same pattern split by fraud label.
SELECT
    isFraud,
    COUNT(*) AS total_transactions,
    SUM(
        CASE
            WHEN amount > 0 AND oldbalanceOrg = 0 AND newbalanceOrig = 0
            THEN 1 ELSE 0
        END
    ) AS zero_balance_transactions,
    ROUND(
        100.0 * AVG(
            CASE
                WHEN amount > 0 AND oldbalanceOrg = 0 AND newbalanceOrig = 0
                THEN 1.0 ELSE 0.0
            END
        ), 2
    ) AS percentage
FROM transactions
GROUP BY isFraud;

-- 09. SUBQUERY: TRANSACTIONS ABOVE OVERALL AVERAGE AMOUNT
SELECT type, amount, isFraud
FROM transactions
WHERE amount > (SELECT AVG(amount) FROM transactions)
ORDER BY amount DESC
LIMIT 20;

-- 10. CASE: TRANSACTION AMOUNT CATEGORIES
-- Thresholds are exploratory categories, not fraud labels.
SELECT
    type,
    amount,
    isFraud,
    CASE
        WHEN amount >= 1000000 THEN 'High Value'
        WHEN amount >= 100000 THEN 'Medium Value'
        ELSE 'Low Value'
    END AS transaction_category
FROM transactions
ORDER BY amount DESC
LIMIT 20;

-- 11. CTE + RANKING: RANK TYPES BY FRAUD COUNT
WITH fraud_summary AS (
    SELECT
        type,
        COUNT(*) AS total_transactions,
        SUM(isFraud) AS fraud_count
    FROM transactions
    GROUP BY type
)
SELECT
    type,
    total_transactions,
    fraud_count,
    RANK() OVER (ORDER BY fraud_count DESC) AS fraud_rank,
    DENSE_RANK() OVER (ORDER BY fraud_count DESC) AS dense_fraud_rank,
    ROW_NUMBER() OVER (ORDER BY fraud_count DESC, type) AS row_number_value
FROM fraud_summary
ORDER BY fraud_rank, type;

-- 12. WINDOW FUNCTION: EACH ROW WITH ITS TYPE AVERAGE
-- LIMIT restricts displayed rows, not the rows used by the window average.
SELECT
    type,
    amount,
    ROUND(AVG(amount) OVER (PARTITION BY type), 2) AS type_average_amount
FROM transactions
LIMIT 20;

-- 13. FINAL INVESTIGATION: TOP 20 HIGH-VALUE TRANSACTIONS
-- Difference = sender balance decrease - transaction amount.
WITH transaction_analysis AS (
    SELECT
        type,
        amount,
        isFraud,
        oldbalanceOrg,
        newbalanceOrig,
        (oldbalanceOrg - newbalanceOrig) - amount AS sender_balance_difference
    FROM transactions
    WHERE amount >= 1000000
)
SELECT
    type,
    amount,
    isFraud,
    oldbalanceOrg,
    newbalanceOrig,
    ROUND(sender_balance_difference, 2) AS sender_balance_difference,
    RANK() OVER (ORDER BY amount DESC) AS amount_rank
FROM transaction_analysis
ORDER BY amount DESC
LIMIT 20;