import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model and preprocessing tools
model = joblib.load("rf_model.pkl")
scaler_x = joblib.load("scaler_x.pkl")
scaler_y = joblib.load("scaler_y.pkl")
oh_encoder = joblib.load("oh_encoder.pkl")

# Streamlit UI
st.set_page_config(page_title="Medical Cost Predictor", page_icon="💊")
st.title("💊 Medical Insurance Cost Predictor")
st.write("Estimate your medical cost based on your personal and lifestyle attributes.")

# User input
age = st.slider("Age", 18, 100, 30)
sex = st.radio("Sex", ["male", "female"])
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
children = st.number_input("Children", 0, 10, 0)
smoker = st.radio("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Prediction function
def predict_cost(input_dict, model):
    df = pd.DataFrame([input_dict])
    df['sex'] = df['sex'].replace({'female': 0, 'male': 1}).astype(int)
    df['smoker'] = df['smoker'].replace({'yes': 1, 'no': 0}).astype(int)
    df['bmi'] = df['bmi'].clip(lower=15, upper=47)

    encoded = oh_encoder.transform(df[['region']])
    encoded_df = pd.DataFrame(encoded.toarray(), columns=['northeast', 'northwest', 'southeast', 'southwest'])
    df = pd.concat([df, encoded_df], axis=1).drop(columns='region')
    
    df[['age', 'bmi']] = scaler_x.transform(df[['age', 'bmi']])
    prediction = model.predict(df)
    
    return scaler_y.inverse_transform(prediction.reshape(-1, 1))[0][0]

# Run prediction
if st.button("Predict Cost"):
    input_data = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }
    cost = predict_cost(input_data, model)
    st.success(f"Estimated Medical Cost: ${cost:,.2f}")

    # Prepare report
    report = (
        "----- Medical Cost Prediction Report -----\n"
        f"Age: {age}\n"
        f"Sex: {sex}\n"
        f"BMI: {bmi}\n"
        f"Children: {children}\n"
        f"Smoker: {smoker}\n"
        f"Region: {region}\n"
        f"------------------------------------------\n"
        f"Estimated Medical Cost: ${cost:,.2f}\n"
    )

    # Download button
    st.download_button(
        label="📥 Download Report",
        data=report,
        file_name="medical_cost_report.txt",
        mime="text/plain"
    )
