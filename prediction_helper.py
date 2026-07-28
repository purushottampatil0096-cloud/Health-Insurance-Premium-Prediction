import os
import pandas as pd
from joblib import load

# ---------------------------------------------------------------------------
# Load artifacts (models + scalers) once, at import time
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ARTIFACTS_DIR = os.path.join(_BASE_DIR, "artifacts")

model_young = load(os.path.join(_ARTIFACTS_DIR, "model_young.joblib"))
model_rest = load(os.path.join(_ARTIFACTS_DIR, "model_rest.joblib"))

scaler_young_data = load(os.path.join(_ARTIFACTS_DIR, "scaler_young.joblib"))
scaler_rest_data = load(os.path.join(_ARTIFACTS_DIR, "scaler_rest.joblib"))

# Final feature order the models expect (taken from model.feature_names_in_)
FEATURE_COLUMNS = [
    "age", "number_of_dependants", "income_level", "income_lakhs",
    "insurance_plan", "genetical_risk", "normalized_risk_score",
    "gender_Male", "region_Northwest", "region_Southeast", "region_Southwest",
    "marital_status_Unmarried", "bmi_category_Obesity",
    "bmi_category_Overweight", "bmi_category_Underweight",
    "smoking_status_Occasional", "smoking_status_Regular",
    "employment_status_Salaried", "employment_status_Self-Employed",
]

# Risk score lookup, exactly as used during training
RISK_SCORES = {
    "diabetes": 6,
    "heart disease": 8,
    "high blood pressure": 6,
    "thyroid": 5,
    "no disease": 0,
    "none": 0,
}
# min/max total_risk_score observed across the full training data
# (2 diseases max: heart disease (8) + diabetes (6) = 14; min is 0)
MIN_RISK_SCORE = 0
MAX_RISK_SCORE = 14


def calculate_normalized_risk(medical_history_1: str, medical_history_2: str) -> float:
    """Combine up to two medical conditions into a normalized risk score (0-1),
    exactly matching the training-time feature engineering."""
    d1 = (medical_history_1 or "none").strip().lower()
    d2 = (medical_history_2 or "none").strip().lower()

    total_risk_score = RISK_SCORES.get(d1, 0) + RISK_SCORES.get(d2, 0)
    normalized_risk_score = (total_risk_score - MIN_RISK_SCORE) / (MAX_RISK_SCORE - MIN_RISK_SCORE)
    return normalized_risk_score


def calculate_income_level(income_lakhs: float) -> int:
    """Bucket raw income (in lakhs) into the same ordinal encoding used in training:
    {'<10L': 1, '10L - 25L': 2, '> 40L': 3, '25L - 40L': 4}"""
    if income_lakhs < 10:
        return 1
    elif income_lakhs < 25:
        return 2
    elif income_lakhs <= 40:
        return 4
    else:
        return 3


def _build_feature_row(
    age,
    number_of_dependants,
    income_lakhs,
    genetical_risk,
    insurance_plan,
    employment_status,
    gender,
    marital_status,
    bmi_category,
    smoking_status,
    region,
    medical_history_1,
    medical_history_2,
):
    """Builds a single-row DataFrame with all 19 engineered features,
    zero-initialised then filled in exactly like the notebooks' pipeline."""

    df = pd.DataFrame(0.0, columns=FEATURE_COLUMNS, index=[0])

    # Numeric / ordinal fields
    df.loc[0, "age"] = age
    df.loc[0, "number_of_dependants"] = number_of_dependants
    df.loc[0, "income_lakhs"] = income_lakhs
    df.loc[0, "genetical_risk"] = genetical_risk
    df.loc[0, "income_level"] = calculate_income_level(income_lakhs)
    df.loc[0, "insurance_plan"] = {"Bronze": 1, "Silver": 2, "Gold": 3}[insurance_plan]
    df.loc[0, "normalized_risk_score"] = calculate_normalized_risk(medical_history_1, medical_history_2)

    # One-hot fields (baseline category left at 0)
    if gender == "Male":
        df.loc[0, "gender_Male"] = 1

    if region == "Northwest":
        df.loc[0, "region_Northwest"] = 1
    elif region == "Southeast":
        df.loc[0, "region_Southeast"] = 1
    elif region == "Southwest":
        df.loc[0, "region_Southwest"] = 1
    # Northeast -> baseline, all zero

    if marital_status == "Unmarried":
        df.loc[0, "marital_status_Unmarried"] = 1

    if bmi_category == "Obesity":
        df.loc[0, "bmi_category_Obesity"] = 1
    elif bmi_category == "Overweight":
        df.loc[0, "bmi_category_Overweight"] = 1
    elif bmi_category == "Underweight":
        df.loc[0, "bmi_category_Underweight"] = 1
    # Normal -> baseline, all zero

    if smoking_status == "Occasional":
        df.loc[0, "smoking_status_Occasional"] = 1
    elif smoking_status == "Regular":
        df.loc[0, "smoking_status_Regular"] = 1
    # No Smoking -> baseline, all zero

    if employment_status == "Salaried":
        df.loc[0, "employment_status_Salaried"] = 1
    elif employment_status == "Self-Employed":
        df.loc[0, "employment_status_Self-Employed"] = 1
    # Freelancer -> baseline, all zero

    return df


def predict(
    age,
    number_of_dependants,
    income_lakhs,
    genetical_risk,
    insurance_plan,
    employment_status,
    gender,
    marital_status,
    bmi_category,
    smoking_status,
    region,
    medical_history_1,
    medical_history_2,
):
    """Full pipeline: builds features, scales them with the correct scaler,
    routes to the correct model (young vs. rest) based on age, and returns
    the predicted annual premium amount, rounded to the nearest integer."""

    df = _build_feature_row(
        age=age,
        number_of_dependants=number_of_dependants,
        income_lakhs=income_lakhs,
        genetical_risk=genetical_risk,
        insurance_plan=insurance_plan,
        employment_status=employment_status,
        gender=gender,
        marital_status=marital_status,
        bmi_category=bmi_category,
        smoking_status=smoking_status,
        region=region,
        medical_history_1=medical_history_1,
        medical_history_2=medical_history_2,
    )

    if age <= 25:
        model = model_young
        scaler_data = scaler_young_data
    else:
        model = model_rest
        scaler_data = scaler_rest_data

    scaler = scaler_data["scaler"]
    cols_to_scale = scaler_data["cols_to_scale"]

    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    prediction = model.predict(df)[0]
    return int(round(prediction))
