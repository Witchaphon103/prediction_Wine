import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# โหลดโมเดลและ scaler
model = joblib.load("./best_wine_quality_model.pkl")
scaler = joblib.load("./scaler.pkl")

# โหลด CSV อัตโนมัติจากโฟลเดอร์
file_path = "./winequality-red.csv"
df = pd.read_csv(file_path)

# ------------------------- UI Styling -------------------------
st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="wide")

# ------------------------- Header -------------------------
st.markdown("<h1>🍷 โปรแกรมทำนายคุณภาพไวน์</h1>", unsafe_allow_html=True)
st.markdown("<h3>🔍 วิเคราะห์คุณภาพของไวน์จากคุณสมบัติทางเคมี</h3>", unsafe_allow_html=True)

# ------------------------- Layout -------------------------
col1, col2 = st.columns([1, 1])  # แบ่งหน้าจอเป็น 2 คอลัมน์ (ซ้าย, ขวา)

# ------------------------- Column 1: แสดงข้อมูล CSV -------------------------
with col1:
    st.markdown("## 📊 ข้อมูลไวน์จากไฟล์ CSV")
    st.dataframe(df.head(10))  # แสดงข้อมูล 10 แถวแรก

    # แสดงสถิติเบื้องต้น
    st.markdown("### 📈 ค่าสถิติเบื้องต้นของข้อมูล")
    st.write(df.describe())

# ------------------------- Column 2: Prediction Form -------------------------
with col2:
    st.markdown("## 🧪 ป้อนค่าคุณสมบัติของไวน์")
    
    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            fixed_acidity = st.number_input("Fixed Acidity (ค่าความเป็นกรดคงที่)", min_value=0.0, step=0.1)
            volatile_acidity = st.number_input("Volatile Acidity (ค่าความเป็นกรดระเหย)", min_value=0.0, step=0.01)
            citric_acid = st.number_input("Citric Acid (ค่ากรดซิตริก)", min_value=0.0, step=0.01)
            residual_sugar = st.number_input("Residual Sugar (ปริมาณน้ำตาลตกค้าง)", min_value=0.0, step=0.1)
            chlorides = st.number_input("Chlorides (ค่าคลอไรด์)", min_value=0.0, step=0.001)
            free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide (ปริมาณซัลเฟอร์ไดออกไซด์อิสระ)", min_value=0.0, step=1.0)

        with col2:
            total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide (ปริมาณซัลเฟอร์ไดออกไซด์ทั้งหมด)", min_value=0.0, step=1.0)
            density = st.number_input("Density (ค่าความหนาแน่น)", min_value=0.0, step=0.0001)
            pH = st.number_input("pH (ค่าพีเอช)", min_value=0.0, step=0.01)
            sulphates = st.number_input("Sulphates (ปริมาณซัลเฟต)", min_value=0.0, step=0.01)
            alcohol = st.number_input("Alcohol (ปริมาณแอลกอฮอล์)", min_value=0.0, step=0.1)

    # ------------------------- Predict Button -------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔮 ทำนายคุณภาพไวน์"):
        # เตรียมข้อมูลสำหรับโมเดล
        input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                                free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])
        input_scaled = scaler.transform(input_data)

        # ทำนาย
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        # กำหนดระดับคะแนนคุณภาพ (1-10)
        quality_score = int(probability[1] * 10)

        # เตรียมข้อความและสี
        if prediction == 1:
            result_text = f"🍷 <b>ไวน์คุณภาพสูง!</b> (ระดับ: {quality_score}/10)"
            result_color = "#90EE90"
            st.balloons()
        else:
            result_text = f"⚠️ <b>ไวน์คุณภาพต่ำ</b> (ระดับ: {10 - quality_score}/10)"
            result_color = "#ff6666"
            st.snow()

        # ------------------------- Show Result -------------------------
        st.markdown(
            f"""
            <div style='padding: 20px; background-color: {result_color}; color: black; border-radius: 10px; text-align: center;'>
                <h2>{result_text}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

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
        <a href="https://line.me/R/ti/p/@777xzgmw" target="_blank">💬 พูดคุยกับ LINE Bot</a>
    </div>
    """,
    unsafe_allow_html=True
)
