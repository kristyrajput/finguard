# FinGuard Pro — SQL Findings Report

## 1. Dataset Overview

- **Dataset:** PaySim synthetic mobile-money transactions
- **Total transactions:** 6,362,620
- **Fraud-labelled transactions:** 8,213
- **Fraud rate:** 0.1291%
- **Existing flagged-fraud rows:** 16
- **Simulation step range:** 1–743

## 2. Fraud by Transaction Type

- TRANSFER fraud rate: 0.7688%
- CASH_OUT fraud rate: 0.1840%
- CASH_IN, DEBIT and PAYMENT have zero fraud-labelled transactions in this dataset.

TRANSFER has the highest observed fraud rate, while CASH_OUT has the largest fraud count.

## 3. High-Value Transactions

Using an exploratory threshold of 1,000,000:

- CASH_OUT: 1,648 transactions, 1,345 fraud-labelled; observed fraud rate 81.61%.
- TRANSFER: 128,859 transactions, 1,361 fraud-labelled; observed fraud rate approximately 1.06%.
- CASH_IN: 119 transactions, no fraud-labelled transactions.

These are dataset-specific observations, not universal fraud rules.

## 4. Balance Analysis

Sender balance matching was approximately 99.45% among fraud-labelled transactions overall, compared with 21.30% among non-fraud transactions.

Receiver balance matching was approximately 48.42% among fraud-labelled transactions and 39.89% among non-fraud transactions.

Patterns differed by transaction type, so balance features should be analysed conditionally by type and validated before model use.

## 5. Zero Sender Balance Pattern

For positive-amount transactions where both old and new sender balances were zero:

- Non-fraud: 32.87%
- Fraud-labelled: 0.30%

This pattern was not a strong positive indicator of fraud in this dataset.

## 6. Limitations

- PaySim is a synthetic dataset; findings may not generalise to real financial transactions.
- Existing fraud labels are used for analysis, not treated as predictions from our own model.
- High transaction value alone does not establish fraud.
- Candidate features require validation during model development.

## 7. Next Steps

1. Continue Python-based exploratory data analysis and feature engineering.
2. Validate candidate features and prevent data leakage.
3. Train and evaluate fraud-classification models using appropriate metrics for imbalanced data.