# FinGuard Pro — Model Evaluation Report

## 1. Project Overview

FinGuard Pro is a financial transaction fraud analytics project built using Python, Pandas, Scikit-learn, and MySQL. The objective of the machine learning component is to identify potentially fraudulent transactions while balancing fraud detection against false alarms.

The project uses the PaySim synthetic mobile-money transaction dataset.

## 2. Dataset Summary

- **Total transactions:** 6,362,620
- **Total features in the original dataset:** 11
- **Fraudulent transactions:** 8,213
- **Non-fraudulent transactions:** 6,354,407
- **Fraud rate:** Approximately 0.1291%
- **Target variable:** `isFraud`

The dataset is highly imbalanced because fraudulent transactions represent a very small proportion of all transactions. Therefore, accuracy alone is not a suitable measure of model performance.

## 3. Feature Engineering

Six additional features were created:

1. `sender_balance_diff` — Difference between the sender's old and new balance.
2. `receiver_balance_diff` — Difference between the receiver's new and old balance.
3. `sender_balance_error` — Difference between the sender balance change and transaction amount.
4. `receiver_balance_error` — Difference between the receiver balance change and transaction amount.
5. `is_zero_sender_balance` — Indicates a positive-amount transaction where both sender balances are zero.
6. `amount_log` — Log-transformed transaction amount using `log1p`.

The categorical `type` feature was converted into numerical indicator columns using one-hot encoding.

## 4. Data Splitting and Evaluation Strategy

For the final Random Forest workflow, the dataset was divided using stratified sampling:

- **Training set:** 60%
- **Validation set:** 20%
- **Test set:** 20%

Stratification maintained approximately the same fraud proportion across the splits.

The training set was used to fit the final Random Forest model. The validation set was used to compare classification thresholds. The test set was reserved for final evaluation.

The Logistic Regression baseline was trained earlier using an 80% training and 20% test split. Both models were evaluated on the same test rows, but their training workflows differed. Consequently, this comparison is useful but is not a perfectly controlled comparison of training strategies.

## 5. Models Evaluated

### 5.1 Logistic Regression

Logistic Regression was used as the baseline classification model.

Configuration:
- Class weighting: `balanced`
- Maximum iterations: 200
- Random state: 42

### 5.2 Random Forest Classifier

Random Forest was used as the final candidate because it performed better than Logistic Regression in the reported test-set comparison.

Configuration:
- Number of trees: 150
- Maximum depth: 16
- Minimum samples per leaf: 5
- Class weighting: `balanced`
- Random state: 42
- Parallel processing: `n_jobs=-1`

The final Random Forest was trained on the full training partition rather than a sampled subset.

## 6. Threshold Selection

A probability threshold of **0.95** was selected provisionally using validation data.

A transaction was classified as potentially fraudulent when its predicted fraud probability was greater than or equal to 0.95.

A higher threshold reduces false alarms but can increase the number of missed fraud cases. The selected threshold represents a trade-off between precision and recall, rather than a universally optimal setting.

## 7. Model Comparison

Both models were evaluated on the same test set containing 1,272,524 transactions, including 1,643 fraudulent transactions.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Threshold | 0.95 | 0.95 |
| Precision | 17.93% | 47.66% |
| Recall | 68.72% | 77.54% |
| F1-score | 0.2845 | 0.5904 |
| PR-AUC (Average Precision) | 0.5670 | 0.7841 |
| ROC-AUC | 0.9922 | 0.9962 |
| True Positives | 1,129 | 1,274 |
| False Positives | 5,166 | 1,399 |
| False Negatives | 514 | 369 |

Random Forest achieved better results across the reported metrics at the selected threshold. It detected more fraudulent transactions while generating fewer false alarms.

## 8. Final Random Forest Test Results

### Confusion Matrix

| | Predicted Non-Fraud | Predicted Fraud |
|---|---:|---:|
| Actual Non-Fraud | 1,269,482 | 1,399 |
| Actual Fraud | 369 | 1,274 |

### Key Metrics

- **Precision:** 47.66%
- **Recall:** 77.54%
- **F1-score:** 0.5904
- **PR-AUC:** 0.7841
- **ROC-AUC:** 0.9962
- **Accuracy:** 99.86%

At the selected threshold, the model flagged 2,673 transactions as potentially fraudulent. Of these, 1,274 were fraudulent according to the dataset labels, while 1,399 were false positives. The model missed 369 of the 1,643 actual fraud cases in the test set.

Although accuracy is high, precision, recall, F1-score, and PR-AUC are more informative for this highly imbalanced classification problem.

## 9. Feature Importance

The Random Forest feature-importance analysis identified the following leading features:

| Feature | Importance |
|---|---:|
| `sender_balance_error` | 0.3970 |
| `sender_balance_diff` | 0.2278 |
| `type_PAYMENT` | 0.0932 |
| `amount_log` | 0.0724 |
| `type_TRANSFER` | 0.0621 |

Sender balance-related features contributed strongly to the model's predictions. However, feature importance indicates predictive contribution within this trained model; it does not establish causation. Related balance features may also share information.

## 10. Limitations

- PaySim is a synthetic dataset, so results may not generalize directly to real banking transactions.
- Fraud cases are extremely rare, making false-positive control important.
- A probability threshold of 0.95 is a provisional operating choice, not a universally optimal threshold.
- The Logistic Regression and Random Forest models used different training splits.
- Predicted probabilities have not been independently calibrated against real-world fraud probabilities.
- Feature importance does not establish causal relationships.
- The current evaluation is offline; it does not demonstrate production performance on live transactions.

## 11. Future Improvements

- Compare models using an identical training and validation methodology.
- Evaluate threshold choices against explicit financial costs of missed fraud and false alarms.
- Investigate probability calibration and temporal validation.
- Build an interactive Power BI dashboard for fraud trends, transaction types, and key performance indicators.
- Develop a transaction-risk prediction API and an investigation workflow.
- Test and document the complete system before deployment.

## 12. Conclusion

The final Random Forest model achieved a fraud recall of 77.54%, precision of 47.66%, F1-score of 0.5904, and PR-AUC of 0.7841 on the held-out test set at a threshold of 0.95.

It outperformed the Logistic Regression baseline in the reported comparison. These findings demonstrate the potential of machine learning for transaction-risk analysis within the PaySim dataset. Further validation on realistic, independently sourced data would be necessary before making claims about real-world banking performance.