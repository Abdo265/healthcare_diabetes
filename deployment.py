import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the best model and scaler
with open('best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("🩺 Diabetes Prediction App")
st.write("This app predicts whether a person has diabetes or not based on the given features.")

age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Gender", ["Male", "Female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0)
bp = st.number_input("Blood Pressure", min_value=60, max_value=200, value=120)
glucose_level = st.number_input("Glucose Level", min_value=50, max_value=300, value=100)
physical_activity = st.selectbox("Physical Activity", ["Low", "Medium", "High"])
family_history = st.selectbox("Family History", ["No", "Yes"])


sex_encoded = 1 if sex == "Male" else 0
BMI_Glucose_Interaction = bmi * glucose_level
High_BP = 1 if bp > 130 else 0

# Create BMI Category
if bmi < 18.5:
    bmi_category = 'Underweight'
elif bmi < 24.9:
    bmi_category = 'Normal'
elif bmi < 29.9:
    bmi_category = 'Overweight'
else:
    bmi_category = 'Obese'


# Prepare input data as a DataFrame to match the training data structure
input_data = pd.DataFrame({
    'Age': [age],
    'BMI': [bmi],
    'BloodPressure': [bp],
    'GlucoseLevel': [glucose_level],
    'BMI_Glucose_Interaction': [BMI_Glucose_Interaction],
    'High_BP': [High_BP],
    'Gender_Male': [sex_encoded],
    'PhysicalActivity_Low': [1 if physical_activity == 'Low' else 0],
    'PhysicalActivity_Medium': [1 if physical_activity == 'Medium' else 0],
    'FamilyHistory_Yes': [1 if family_history == 'Yes' else 0],
    'BMI_Category_Normal': [1 if bmi_category == 'Normal' else 0],
    'BMI_Category_Overweight': [1 if bmi_category == 'Overweight' else 0],
    'BMI_Category_Obese': [1 if bmi_category == 'Obese' else 0]
})

# Scale the numerical features
numerical_features = ['Age', 'BMI', 'BloodPressure', 'GlucoseLevel', 'BMI_Glucose_Interaction', 'High_BP']
input_data[numerical_features] = scaler.transform(input_data[numerical_features])


if st.button("Predict"):
    prediction = best_model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ The person is likely to have diabetes (والعياذ بالله).")
    else:
        st.success("✅ The person is unlikely to have diabetes (والحمد لله).")