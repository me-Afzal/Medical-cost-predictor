import streamlit as st
import pandas as pd
import numpy as np
import joblib
from fpdf import FPDF
import tempfile
import os

# Load saved model and preprocessing tools
model = joblib.load("rf_model.pkl")
scaler_x = joblib.load("scaler_x.pkl")
scaler_y = joblib.load("scaler_y.pkl")
oh_encoder = joblib.load("oh_encoder.pkl")

# Streamlit UI
st.set_page_config(page_title="Medical Cost Predictor", page_icon="💊")
st.title("💊 Medical Insurance Cost Predictor")
st.write("Estimate your medical cost based on your personal and lifestyle attributes.")

# User details
st.header("👤 User Information")
name = st.text_input("Full Name")
email = st.text_input("Email Address")
phone = st.text_input("Phone Number")

# User input for prediction
st.header("📋 Medical Details")
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

# PDF report function with centered logo
def generate_pdf(user_info, input_data, cost):
    pdf = FPDF()
    pdf.add_page()

    # Centered logo
    logo_path = "logo.png"  # Make sure logo.png is in the same directory
    if os.path.exists(logo_path):
        image_width = 40  # in mm
        page_width = pdf.w
        center_x = (page_width - image_width) / 2
        pdf.image(logo_path, x=center_x, y=10, w=image_width)

    pdf.ln(35)  # space after logo
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt="Medical Insurance Cost Report", ln=True, align='C')
    pdf.ln(10)

    # User details
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, txt=f"Name: {user_info['name']}", ln=True)
    pdf.cell(0, 10, txt=f"Email: {user_info['email']}", ln=True)
    pdf.cell(0, 10, txt=f"Phone: {user_info['phone']}", ln=True)
    pdf.ln(10)

    # Input details
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="Input Details:", ln=True)
    pdf.set_font("Arial", '', 12)
    for key, value in input_data.items():
        pdf.cell(0, 10, txt=f"{key.capitalize()}: {value}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"Estimated Medical Cost: ${cost:,.2f}", ln=True)

    # Save temp PDF
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)
    return temp_file.name

# Run prediction
if st.button("Predict Cost"):
    if not name or not email or not phone:
        st.warning("Please fill in your personal information.")
    else:
        input_data = {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region
        }
        user_info = {
            "name": name,
            "email": email,
            "phone": phone
        }

        cost = predict_cost(input_data, model)
        st.success(f"Estimated Medical Cost: ${cost:,.2f}")

        pdf_path = generate_pdf(user_info, input_data, cost)
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📄 Download PDF Report",
                data=f,
                file_name="medical_cost_report.pdf",
                mime="application/pdf"
            )
