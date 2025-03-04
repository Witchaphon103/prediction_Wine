import streamlit as st
import numpy as np
import requests  # ใช้สำหรับเรียก API

# 🔹 ใส่ URL ของ Flask API ที่ Deploy บน Railway
API_URL = "https://predictionwine-production.up.railway.app/predict"

# ------------------------- UI Styling -------------------------
st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="centered")

# CSS Styling - เพิ่ม Animation, Glow, Confetti/Snow, ปรับพื้นหลัง + ปรับแถบ Header
st.markdown(
    """
    <style>
        [data-testid="stHeader"] {
            background: linear-gradient(to right, #1e1e1e, #3a3a3a) !important;
            color: #fff !important;
            box-shadow: none !important;
            border-bottom: 1px solid #444 !important;
        }
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(to right, #1e1e1e, #3a3a3a) !important;
            color: white;
        }
        .stButton>button {
            background: linear-gradient(45deg, #800020, #b22222);
            color: white !important;
            border-radius: 15px;
            padding: 16px 30px;
            font-size: 20px;
            font-weight: bold;
            transition: 0.3s ease-in-out;
            box-shadow: 0px 5px 15px rgba(255, 100, 100, 0.3);
        }
        .stButton>button:hover {
            background: linear-gradient(45deg, #b22222, #ff4747);
            transform: scale(1.08);
            box-shadow: 0px 5px 25px rgba(255, 100, 100, 0.5);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------- Header -------------------------
st.markdown("<h1>🍷 Wine Quality Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3>🔍 กรอกค่าคุณสมบัติเพื่อทำนายคุณภาพของไวน์</h3>", unsafe_allow_html=True)

# ------------------------- Input Fields -------------------------
st.markdown("<h2>📊 ค่าคุณสมบัติของไวน์</h2>", unsafe_allow_html=True)
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
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🔮 Predict Wine Quality"):
    # 🔹 ส่งข้อมูลไปยัง API
    input_data = {
        "features": [
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
            free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol
        ]
    }
    
    response = requests.post(API_URL, json=input_data)
    
    # 🔹 ตรวจสอบว่าการเรียก API สำเร็จหรือไม่
    if response.status_code == 200:
        result = response.json()
        prediction = result.get("prediction", 0)
        quality_score = result.get("quality_score", 5)  # ถ้าไม่มีค่าจะ Default เป็น 5

        # เตรียมข้อความและสี
        if prediction == 1:
            result_text = f"🍷 <b>ไวน์คุณภาพสูง!</b> (ระดับ: {quality_score}/10)"
            result_color = "#90EE90"
            st.balloons()  # แสดงลูกโป่งถ้าไวน์คุณภาพสูง
        else:
            result_text = f"⚠️ <b>ไวน์คุณภาพต่ำ</b> (ระดับ: {10 - quality_score}/10)"
            result_color = "#ff6666"
            st.snow()      # แสดงหิมะเอฟเฟกต์ถ้าไวน์คุณภาพต่ำ

        # ------------------------- Show Result -------------------------
        st.markdown(
            f"""
            <div style='background: rgba(50, 50, 50, 0.9); padding: 20px; border-radius: 15px; text-align: center;'>
                <h2 style='color: {result_color};'>{result_text}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error("❌ มีข้อผิดพลาดในการทำนาย กรุณาลองใหม่")

# ------------------------- LINE Chatbot Button -------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
    <style>
        .line-button {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
        .line-button a {
            background-color: #06C755;
            color: white;
            padding: 15px 30px;
            font-size: 20px;
            font-weight: bold;
            text-decoration: none;
            border-radius: 10px;
            transition: 0.3s ease-in-out;
            display: inline-block;
        }
        .line-button a:hover {
            background-color: #04a145;
            transform: scale(1.05);
        }
    </style>
    <div class="line-button">
        <a href="https://line.me/R/ti/p/@777xzgmw" target="_blank">💬 Chat กับ LINE Bot</a>
    </div>
    """,
    unsafe_allow_html=True
)
