# Healthcare Premium Prediction

A machine learning web app that predicts a customer's **annual health insurance premium** based on demographic, lifestyle, and medical information — built with segmented regression models and deployed as an interactive Streamlit app.

---

## Overview

Insurance premiums depend on a mix of factors — age, income, medical history, lifestyle habits — that interact differently across age groups. Rather than fitting one model to the entire population, this project **segments customers into two age bands** and trains a separate model for each, since risk factors (like genetic risk) behave very differently for young vs. older customers.

- **Age ≤ 25** → Linear Regression model
- **Age > 25** → Tuned XGBoost Regressor (via `RandomizedSearchCV`)

The final models are wrapped in a clean Streamlit UI where a user fills in a form and instantly gets a predicted annual premium.

---

## Features

- End-to-end ML pipeline: data cleaning → outlier handling → feature engineering → encoding → scaling → model training → evaluation
- Age-based model segmentation for improved accuracy over a single global model
- Custom feature engineering:
  - **Risk score** derived from combinations of up to two medical conditions, normalized to a 0–1 scale
  - **Income level** bucketed from raw income for ordinal encoding
- Multicollinearity checked via **VIF (Variance Inflation Factor)**
- Model comparison (Linear Regression, Ridge, XGBoost) with hyperparameter tuning
- Error analysis on residuals to catch systematic under/over-prediction
- Interactive **Streamlit** frontend for real-time predictions

---

## Methodology

1. **Data Segmentation** — the raw dataset (50,000 records) is split into two segments by age (`≤ 25` and `> 25`), since younger and older policyholders show different premium behavior.
2. **Data Cleaning** — handled missing values, negative dependants, unrealistic ages (>100), and extreme income outliers (99.9th percentile cutoff).
3. **Feature Engineering**
   - Combined `Medical History` (up to 2 conditions) into a single `normalized_risk_score` using a weighted severity scale.
   - Encoded `Insurance_Plan` and `Income_Level` as ordinal features.
   - One-hot encoded nominal categorical features (`Gender`, `Region`, `Marital_status`, `BMI_Category`, `Smoking_Status`, `Employment_Status`).
4. **Scaling** — numeric features scaled with `MinMaxScaler` (fit separately per segment).
5. **Model Training & Selection**
   - Young segment: Linear Regression performed best.
   - Rest segment: XGBoost (tuned) outperformed Linear Regression / Ridge.
6. **Evaluation** — R², MSE, RMSE, and residual/error-percentage analysis to flag high-error predictions.
7. **Deployment** — models and scalers serialized with `joblib`, served through a Streamlit app.

---

## 🗂️ Project Structure

```
Healthcare Premium Prediction/
│
├── app/
│   ├── main.py                  # Streamlit UI
│   ├── prediction_helper.py     # Feature engineering + prediction logic
│   └── artifacts/
│       ├── model_young.joblib
│       ├── model_rest.joblib
│       ├── scaler_young.joblib
│       └── scaler_rest.joblib
│
├── artifacts/                   # Same model/scaler files (used by the notebooks)
│   ├── model_young.joblib
│   ├── model_rest.joblib
│   ├── scaler_young.joblib
│   └── scaler_rest.joblib
│
├── data_segmentation.ipynb                  # Splits raw data into young / rest segments
├── ml_premium_prediction.ipynb              # Baseline EDA + modeling on full dataset
├── ml_premium_prediction_young.ipynb        # Modeling for age ≤ 25 segment
├── ml_premium_prediction_young_with_gr.ipynb # Adds genetical risk feature (young)
├── ml_premium_prediction_rest.ipynb         # Modeling for age > 25 segment
├── ml_premium_prediction_rest_with_gr.ipynb # Adds genetical risk feature (rest)
│
├── premiums.xlsx                # Full raw dataset
├── premiums_young.xlsx          # Age ≤ 25 subset
├── premiums_rest.xlsx           # Age > 25 subset
├── premiums_young_with_gr.xlsx  # Age ≤ 25 subset with genetical risk column
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

- **50,000** customer records with **13 features**: Age, Gender, Region, Marital Status, Number of Dependants, BMI Category, Smoking Status, Employment Status, Income Level, Income (Lakhs), Medical History, Insurance Plan, and the target — Annual Premium Amount.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Modeling | scikit-learn (Linear Regression, Ridge), XGBoost |
| Stats | statsmodels (VIF) |
| Serialization | joblib |
| App / Deployment | Streamlit |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/healthcare-premium-prediction.git
cd healthcare-premium-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
cd app
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📈 Model Performance

| Segment | Best Model | Notes |
|---|---|---|
| Age ≤ 25 | Linear Regression | Simpler relationships; linear model generalized best |
| Age > 25 | XGBoost (tuned) | Captured non-linear interactions better than linear models |

*(Add your exact R² / RMSE numbers here from your notebook output if you'd like them displayed.)*

---

## 🔮 Future Improvements

- Add SHAP-based explainability for individual predictions
- Deploy on Streamlit Community Cloud / Render for public access
- Expand medical history handling beyond 2 conditions
- Add authentication and premium-quote history for returning users

---

## 👤 Author

**Purushottam**
📍 Pune, India

If you found this useful, consider ⭐ starring the repo!

---

## 📄 License

This project is licensed under the MIT License — feel free to use and modify it.
