# \# Customer Churn Prediction

# 

# End-to-end machine learning project predicting telecom customer churn. Built as a portfolio piece demonstrating the full ML pipeline: EDA, feature engineering, model comparison, hyperparameter tuning, and business-driven threshold optimization.

# 

# \## Problem

# 

# Telecom companies lose \~25% of customers annually. Acquiring a new customer costs 5-10x more than retaining an existing one. The business question:

# 

# > \*\*Can we predict which customers are likely to churn, so the retention team can proactively engage them?\*\*

# 

# The model identifies high-risk customers, allowing targeted retention campaigns instead of reactive damage control.

# 

# \## Dataset

# 

# \[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample dataset on Kaggle).

# 

# \- 7,043 customers

# \- 21 features: demographics, account information, services subscribed

# \- Target: `Churn` (Yes/No)

# \- Class balance: 73.5% retained, 26.5% churned (mild imbalance)

# 

# \## Project Structure



```

customer-churn-prediction/

├── data/

│   ├── WA\_Fn-UseC\_-Telco-Customer-Churn.csv   # raw data (gitignored)

│   └── processed/                              # processed train/test (gitignored)

├── notebooks/

│   ├── 01\_eda.ipynb                            # exploratory data analysis

│   ├── 02\_feature\_engineering.ipynb            # encoding, scaling, splits

│   ├── 03\_modeling.ipynb                       # LogReg, RF, XGBoost comparison

│   └── 04\_final\_model.ipynb                    # CV, tuning, threshold optimization

├── models/                                     # saved model (gitignored)

├── .gitignore

└── README.md

```



\## Approach



\### 1. Exploratory Data Analysis (`01\_eda.ipynb`)

\- Distribution analysis of numerical and categorical features

\- Class imbalance assessment

\- Identified key churn drivers: contract type, tenure, internet service type, payment method



\### 2. Feature Engineering (`02\_feature\_engineering.ipynb`)

\- Dropped non-predictive columns (`customerID`, `gender`)

\- Binary encoding for Yes/No features

\- One-Hot Encoding for nominal categorical features (with `drop\_first=True` to avoid multicollinearity)

\- \*\*Stratified\*\* train/test split (80/20) preserving class distribution

\- StandardScaler fit on training data only (avoiding data leakage)



\### 3. Modeling (`03\_modeling.ipynb`)

Trained and compared three models with `class\_weight='balanced'` to handle imbalance:



| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |

|---|---|---|---|---|---|

| Logistic Regression | 0.737 | 0.503 | \*\*0.786\*\* | 0.614 | \*\*0.842\*\* |

| Random Forest | 0.765 | 0.543 | 0.727 | 0.622 | 0.842 |

| XGBoost | 0.751 | 0.523 | 0.703 | 0.600 | 0.830 |



\*\*Selected: Logistic Regression\*\* — highest recall (critical for churn) and interpretable coefficients for business communication.



\### 4. Final Model (`04\_final\_model.ipynb`)

\- 5-fold Stratified Cross-Validation confirmed model stability (ROC-AUC: 0.846 ± 0.013)

\- GridSearchCV for hyperparameter tuning (best: `C=10`, L2 penalty)

\- \*\*Business-driven threshold optimization\*\*: optimized for expected business value, not F1



\### Why a custom threshold?



The default 0.5 threshold treats false positives and false negatives equally. In churn prediction, costs are asymmetric:



\- \*\*Missed churner\*\*: \~1000 PLN in lost customer lifetime value

\- \*\*False alarm\*\*: \~15 PLN for an unnecessary retention call



Optimizing for expected business value yielded an optimal threshold of \*\*0.07\*\*, resulting in:

\- \*\*Recall: 99.5%\*\* — catches nearly all churners

\- \*\*Precision: 32.1%\*\* — more false alarms, but they're cheap

\- \*\*Expected business value: +358,180 PLN\*\* on the test set



\## Key Insights



Feature importance from Random Forest confirms EDA findings:



1\. \*\*Tenure\*\* — newest customers churn most. Loyalty grows with time.

2\. \*\*Contract type\*\* — month-to-month contracts churn at 43% vs 3% for two-year contracts.

3\. \*\*Fiber optic internet\*\* — surprisingly high churn (42%), suggesting service quality or pricing issues.

4\. \*\*Electronic check\*\* payments correlate with churn — these customers are less "locked in".

5\. \*\*Add-on services\*\* (OnlineSecurity, TechSupport) reduce churn — each subscription increases stickiness.



\## Tech Stack



\- \*\*Python 3.12\*\*

\- \*\*pandas, NumPy\*\* — data manipulation

\- \*\*scikit-learn\*\* — preprocessing, models, evaluation, hyperparameter tuning

\- \*\*XGBoost\*\* — gradient boosting

\- \*\*matplotlib, seaborn\*\* — visualization

\- \*\*Jupyter\*\* — interactive development

\- \*\*joblib\*\* — model serialization



\## How to Run



1\. Clone the repo:

```bash

git clone https://github.com/bzietek/customer-churn-prediction.git

cd customer-churn-prediction

```



2\. Create a virtual environment and install dependencies:

```bash

python -m venv venv

venv\\Scripts\\activate  # Windows

pip install -r requirements.txt

```



3\. Download the dataset from \[Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in `data/`.



4\. Run notebooks in order:

```bash

jupyter notebook

\# Open notebooks/01\_eda.ipynb and run each notebook sequentially

```



\## Next Steps



\- \*\*A/B testing\*\* in production against the current retention process

\- \*\*Model monitoring\*\* for concept drift over time

\- \*\*Quarterly retraining\*\* with fresh data

\- \*\*Cost parameter refinement\*\* with business stakeholders

\- \*\*Explore deep learning approaches\*\* (e.g., TabNet) for performance comparison

\- \*\*Build an API wrapper\*\* (FastAPI) for real-time predictions



\## Author



\*\*Bartosz Ziętek\*\*

\[LinkedIn](https://www.linkedin.com/in/bartosz-zietek) | \[GitHub](https://github.com/bzietek)



MSc Data Science and Machine Learning, University of Silesia

AI Research Associate at Keywords Studios



\## License



MIT

