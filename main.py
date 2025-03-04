import streamlit as st
import numpy as np
import requests

# ------------------------- UI -------------------------
st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="centered")

st.markdown("<h1>🍷 Wine Quality Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3>🔍 กรอกค่าคุณสมบัติเพื่อทำนายคุณภาพของไวน์</h3>", unsafe_allow_html=True)

# ------------------------- Input Fields -------------------------
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, step=0.1)
        volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, step=0.01)
        citric_acid = st.number_input("Citric Acid", min_value=0.0, step=0.01)
        residual_sugar = st.number_input("Residual Sugar", min_value=0.0, step=0.1)
        chlorides = st.number_input("Chloride", min_value=0.0, step=0.001)
        free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, step=1.0)

    with col2:
        total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, step=1.0)
        density = st.number_input("Density", min_value=0.0, step=0.0001)
        pH = st.number_input("pH", min_value=0.0, step=0.01)
        sulphates = st.number_input("Sulphates", min_value=0.0, step=0.01)
        alcohol = st.number_input("Alcohol", min_value=0.0, step=0.1)

# ------------------------- Predict Button -------------------------
if st.button("🔮 Predict Wine Quality"):
    # ส่งข้อมูลไปที่ Flask API
    features = [
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
        free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol
    ]

    try:
        response = requests.post("https://your-api-url.up.railway.app/predict", json={"features": features})
        result = response.json()

        # แสดงผลลัพธ์
        if "prediction" in result:
            prediction = result["prediction"]
            st.success(f"🔮 คุณภาพไวน์: {'ดี' if prediction == 1 else 'แย่'}")
        else:
            st.error("❌ มีข้อผิดพลาดในการทำนาย")

    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
