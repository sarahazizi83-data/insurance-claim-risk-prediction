# insurance-claim-risk-prediction
Machine learning project for automobile insurance claim prediction using data mining, Decision Tree, SVM, and feature importance analysis.
# Insurance Claim Risk Prediction with Data Mining

This repository contains a clean, portfolio-ready machine learning project for **predicting automobile insurance claim outcomes** using data mining techniques in Python.

The project demonstrates an end-to-end workflow: data cleaning, exploratory data analysis, preprocessing, model training, model comparison, hyperparameter tuning, and feature importance analysis.

> **Note**: The raw dataset is not included in this repository. Place your CSV file in `data/raw/` before running the code.

---

## Business Problem

Insurance companies need to identify policyholders with a higher probability of filing a claim. A data-driven claim prediction model can support:

- Risk-based underwriting
- Customer segmentation
- Pricing and premium adjustment
- Fraud/risk monitoring
- Better portfolio-level decision making

The target variable is `OUTCOME`:

- `0`: No claim
- `1`: Claim

---

## Project Workflow

1. Load the automobile insurance dataset
2. Remove non-informative identifier columns such as `ID`
3. Inspect missing values and class imbalance
4. Build leakage-safe preprocessing pipelines
5. Encode categorical variables using One-Hot Encoding
6. Train and compare:
   - Decision Tree Classifier
   - Linear Support Vector Classifier
7. Tune SVM hyperparameters using GridSearchCV
8. Evaluate models using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices
9. Extract and visualize feature importance from the Decision Tree model

---

## Why This Version Is Portfolio-Ready

The original notebook was restructured into a professional GitHub format. The revised implementation avoids common modeling mistakes by:

- Removing local machine paths
- Keeping the dataset outside the public repository
- Splitting train/test data before model fitting
- Applying preprocessing inside scikit-learn pipelines
- Avoiding target leakage from outcome-based imputation
- Handling class imbalance with `class_weight='balanced'` instead of oversampling before train/test split
- Saving results, tables, and figures into organized folders

---

## Repository Structure

```text
insurance-claim-risk-mining/
│
├── data/
│   ├── raw/                  # Put the original CSV file here
│   └── processed/            # Optional cleaned outputs
│
├── notebooks/
│   └── insurance_claim_modeling_template.ipynb
│
├── src/
│   └── insurance_claim_modeling.py
│
├── reports/
│   ├── figures/              # Confusion matrices and feature importance plots
│   ├── tables/               # Model metrics and feature importance tables
│   └── insurance-claim-data-mining-report-fa.pdf
│
├── docs/
│   └── modeling-notes.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Expected Input File

Place your dataset here:

```text
data/raw/car_insurance_claim.csv
```

The script expects a CSV file containing a target column named:

```text
OUTCOME
```

Common columns used in the original project include:

```text
ID, AGE, GENDER, RACE, DRIVING_EXPERIENCE, EDUCATION, INCOME,
CREDIT_SCORE, VEHICLE_OWNERSHIP, VEHICLE_YEAR, MARRIED,
CHILDREN, POSTAL_CODE, ANNUAL_MILEAGE, VEHICLE_TYPE,
SPEEDING_VIOLATIONS, DUIS, PAST_ACCIDENTS, OUTCOME
```

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the modeling pipeline:

```bash
python src/insurance_claim_modeling.py --data-path data/raw/car_insurance_claim.csv
```

Outputs will be saved in:

```text
reports/tables/
reports/figures/
```

---

## Main Outputs

The script generates:

- `model_metrics.csv`
- `feature_importance_decision_tree.csv`
- `confusion_matrix_decision_tree.png`
- `confusion_matrix_linear_svm.png`
- `top_features_decision_tree.png`

---

## Key Analytical Insights

The original analysis found that Decision Tree performed better than SVM for claim prediction, while feature importance analysis highlighted variables such as credit score, speeding violations, and annual mileage as influential predictors of insurance claim outcomes.

---

## Skills Demonstrated

- Python data analysis
- Data cleaning and preprocessing
- Missing value handling
- Categorical encoding
- Classification modeling
- Decision Tree
- Support Vector Machine
- GridSearchCV
- Model evaluation
- Feature importance analysis
- Risk analytics for insurance

---

## Disclaimer

This repository is intended for portfolio and educational purposes. No confidential company data should be uploaded. If using a private or proprietary dataset, upload only anonymized samples or keep the dataset outside GitHub.
