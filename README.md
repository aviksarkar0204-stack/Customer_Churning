# Customer Churn Predictor

A machine learning project that predicts customer churn for a telecom company using the IBM Telco Customer Churn dataset. The project covers the full ML pipeline — from EDA and preprocessing to model training, evaluation, and a Streamlit web application for batch predictions.

---

## Demo

Upload a CSV of customer data and get instant churn predictions with probabilities and a downloadable results file.

---

## Project Structure

```
customer-churn-predictor/
│
├── Data_Visualization.ipynb       # EDA and preprocessing notebook
├── data_training.ipynb            # Model training and evaluation notebook
├── app.py                         # Streamlit web application
│
├── Logistic_model.pkl             # Saved Logistic Regression model
├── scaler.pkl                     # Saved StandardScaler
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset
├── Customer_Churn_Data_V2.csv             # Cleaned and encoded dataset
│
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## Dataset

- **Source**: [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size**: 7,043 rows × 21 columns (7,032 after cleaning)
- **Target**: `Churn` — whether a customer left the company (Yes/No)
- **Features**: Demographics, account info, and subscribed services

---

## ML Pipeline

### 1. Exploratory Data Analysis
- Countplots for all categorical features
- Boxplots for outlier detection on `tenure`, `MonthlyCharges`, `TotalCharges`
- Correlation heatmap on numerical features
- `TotalCharges` converted from object to numeric (11 null rows dropped)

### 2. Preprocessing
- **Label Encoding** for binary columns: `gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `Churn`
- **One-Hot Encoding** for multi-class columns: `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `PaymentMethod`, `Contract`, `MultipleLines`
- `customerID` dropped (not a predictive feature)

### 3. Handling Class Imbalance
- Target distribution: ~73% No Churn, ~27% Churn
- **SMOTE** (Synthetic Minority Oversampling Technique) applied on training data only to prevent data leakage

### 4. Train-Test Split
- 80% training, 20% test
- `random_state=42` for reproducibility
- StandardScaler fitted on training data, applied to test data

### 5. Models Trained

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.737 | 0.503 | **0.794** | 0.616 | **0.832** |
| Random Forest (tuned) | 0.760 | 0.537 | 0.693 | 0.605 | 0.826 |
| XGBoost (tuned) | 0.721 | 0.483 | 0.717 | 0.577 | 0.814 |

### 6. Hyperparameter Tuning
- `RandomizedSearchCV` with 5-fold cross validation
- Random Forest tuned on `recall` scoring
- XGBoost tuned on `f1` scoring (recall-only tuning caused precision collapse)

### 7. Final Model
**Logistic Regression** selected as the final model based on:
- Highest recall (0.794) — catches the most churners
- Best AUC (0.832)
- For churn detection, missing a churner is more costly than a false alarm

---

## Key Insights from Feature Importance

Based on Random Forest feature importances:

1. **Contract type (Month-to-month)** — strongest predictor. Monthly customers churn the most due to lack of commitment.
2. **Tenure** — newer customers churn more. Long-term customers are more loyal.
3. **TechSupport_No** — customers without tech support are more likely to leave.
4. **TotalCharges** — correlated with tenure; higher charges indicate longer-staying customers.
5. **OnlineSecurity_No** — lack of security services increases churn risk.

**Business recommendation**: Focus retention efforts on month-to-month customers with low tenure. Bundle tech support and online security to reduce churn.

---

## Streamlit App

The app accepts a CSV upload of customer data and returns:
- Churn prediction (Yes/No) per customer
- Churn probability score
- Summary metrics (total customers, predicted churners, churn rate)
- Downloadable results CSV

### Running the App

```bash
# Clone the repository
git clone https://github.com/aviksarkar0204-stack/customer-churn-predictor.git
cd customer-churn-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### CSV Format for Upload

The uploaded CSV must have exactly these 40 columns in this order, all numeric:

```
gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, PaperlessBilling,
MonthlyCharges, TotalCharges, InternetService_DSL, InternetService_Fiber optic,
InternetService_No, OnlineSecurity_No, OnlineSecurity_No internet service,
OnlineSecurity_Yes, OnlineBackup_No, OnlineBackup_No internet service, OnlineBackup_Yes,
DeviceProtection_No, DeviceProtection_No internet service, DeviceProtection_Yes,
TechSupport_No, TechSupport_No internet service, TechSupport_Yes, StreamingTV_No,
StreamingTV_No internet service, StreamingTV_Yes, StreamingMovies_No,
StreamingMovies_No internet service, StreamingMovies_Yes,
PaymentMethod_Bank transfer (automatic), PaymentMethod_Credit card (automatic),
PaymentMethod_Electronic check, PaymentMethod_Mailed check, Contract_Month-to-month,
Contract_One year, Contract_Two year, MultipleLines_No, MultipleLines_No phone service,
MultipleLines_Yes
```

---

## Tech Stack

- **Language**: Python 3.11
- **ML**: scikit-learn, XGBoost, imbalanced-learn
- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Deployment**: Streamlit
- **Environment**: Conda (condaenv3)

---

## Author

**Avik Sarkar**
B.Tech CSE (AI/ML) — Brainware University Barasat (2024–2028)
GitHub: [aviksarkar0204-stack](https://github.com/aviksarkar0204-stack)
Hugging Face: [Avik128](https://huggingface.co/Avik128)
