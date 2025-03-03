import streamlit as st
import numpy as np
import joblib

# โหลดโมเดลและ scaler
model = joblib.load("./best_wine_quality_model.pkl")
scaler = joblib.load("./scaler.pkl")

# UI ของเว็บแอป
st.title("Wine Quality Predictor")
st.write("กรุณากรอกค่าคุณสมบัติเพื่อทำนายคุณภาพของไวน์")

# ฟอร์มกรอกข้อมูล
fixed_acidity = st.number_input("Fixed Acidity (ความเป็นกรดคงที่)", min_value=0.0, step=0.1)
volatile_acidity = st.number_input("Volatile Acidity (ความเป็นกรดระเหย)", min_value=0.0, step=0.01)
citric_acid = st.number_input("Citric Acid (กรดซิตริก)", min_value=0.0, step=0.01)
residual_sugar = st.number_input("Residual Sugar (น้ำตาลตกค้าง)", min_value=0.0, step=0.1)
chlorides = st.number_input("Chloride (คลอไรด์)", min_value=0.0, step=0.001)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide (ซัลเฟอร์ไดออกไซด์ฟรี)", min_value=0.0, step=1.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide (ซัลเฟอร์ไดออกไซด์ทั้งหมด)", min_value=0.0, step=1.0)
density = st.number_input("density (ความหนาแน่น)", min_value=0.0, step=0.0001)
pH = st.number_input("pH (ค่า pH)", min_value=0.0, step=0.01)
sulphates = st.number_input("Sulphates (ซัลเฟต)", min_value=0.0, step=0.01)
alcohol = st.number_input("Alcohol (แอลกอฮอล์)", min_value=0.0, step=0.1)

# ปุ่มทำนาย
if st.button("Predict Wine Quality"):
    # เตรียมข้อมูลสำหรับโมเดล
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                            free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])
    input_scaled = scaler.transform(input_data)
    
    # ทำนาย
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    # กำหนดระดับคะแนนคุณภาพ (1-10)
    quality_score = int(probability[1] * 10)  # แปลงโอกาสเป็นสเกล 1-10
    
    if prediction == 1:
        result = f"คุณภาพสูง (ระดับ: {quality_score}/10)"
    else:
        result = f"คุณภาพต่ำ (ระดับ: {10 - quality_score}/10)"
    
    st.subheader(f"ผลลัพธ์: {result}")