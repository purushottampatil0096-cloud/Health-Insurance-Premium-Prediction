import streamlit as st
from prediction_helper import predict

st.set_page_config(page_title="Healthcare Premium Prediction", page_icon="🩺", layout="centered")

st.title("🩺 Healthcare Premium Prediction")
st.write(
    "Fill in the details below to get an estimated **Annual Premium Amount** "
    "for a health insurance policy."
)

# --- Options (exactly as seen in the training data) -------------------------
GENDER_OPTIONS = ["Male", "Female"]
REGION_OPTIONS = ["Northeast", "Northwest", "Southeast", "Southwest"]
MARITAL_STATUS_OPTIONS = ["Married", "Unmarried"]
BMI_OPTIONS = ["Normal", "Obesity", "Overweight", "Underweight"]
SMOKING_OPTIONS = ["No Smoking", "Occasional", "Regular"]
EMPLOYMENT_OPTIONS = ["Salaried", "Self-Employed", "Freelancer"]
INSURANCE_PLAN_OPTIONS = ["Bronze", "Silver", "Gold"]
MEDICAL_HISTORY_OPTIONS = [
    "No Disease", "Diabetes", "High blood pressure", "Heart disease", "Thyroid"
]

row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
row4 = st.columns(3)

with row1[0]:
    age = st.number_input("Age", min_value=18, max_value=100, step=1, value=30)
with row1[1]:
    number_of_dependants = st.number_input("Number of Dependants", min_value=0, max_value=20, step=1, value=0)
with row1[2]:
    income_lakhs = st.number_input("Income (in Lakhs)", min_value=1, max_value=200, step=1, value=10)

with row2[0]:
    genetical_risk = st.number_input("Genetical Risk (0-5)", min_value=0, max_value=5, step=1, value=0)
with row2[1]:
    insurance_plan = st.selectbox("Insurance Plan", INSURANCE_PLAN_OPTIONS)
with row2[2]:
    employment_status = st.selectbox("Employment Status", EMPLOYMENT_OPTIONS)

with row3[0]:
    gender = st.selectbox("Gender", GENDER_OPTIONS)
with row3[1]:
    marital_status = st.selectbox("Marital Status", MARITAL_STATUS_OPTIONS)
with row3[2]:
    bmi_category = st.selectbox("BMI Category", BMI_OPTIONS)

with row4[0]:
    smoking_status = st.selectbox("Smoking Status", SMOKING_OPTIONS)
with row4[1]:
    region = st.selectbox("Region", REGION_OPTIONS)
with row4[2]:
    medical_history_1 = st.selectbox("Medical History 1", MEDICAL_HISTORY_OPTIONS)

medical_history_2 = st.selectbox(
    "Medical History 2 (optional second condition)",
    ["None"] + [m for m in MEDICAL_HISTORY_OPTIONS if m != "No Disease"],
)
if medical_history_2 == "None":
    medical_history_2 = "No Disease"

st.markdown("---")

if st.button("Predict Annual Premium", type="primary"):
    prediction = predict(
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
    st.success(f"### Predicted Annual Premium: ₹ {prediction:,}")
    # st.caption(
    #     f"Prediction generated using the "
    #     f"{'**Young (age ≤ 25)**' if age <= 25 else '**Rest (age > 25)**'} model."
    # )
