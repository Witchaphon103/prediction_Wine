import streamlit as st
import numpy as np
import joblib

# โหลดโมเดลและ scaler
model = joblib.load("./best_wine_quality_model.pkl")
scaler = joblib.load("./scaler.pkl")

# ------------------------- UI Styling -------------------------
st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="centered")

# CSS Styling
st.markdown(
    """
    <style>
        /* ปรับสีของแถบเมนูด้านบน (Header) */
        [data-testid="stHeader"] {
            background: linear-gradient(to right, #1e1e1e, #3a3a3a) !important;
            color: #fff !important;
            box-shadow: none !important;
            border-bottom: 1px solid #444 !important;
        }

        /* กำหนดพื้นหลังของ App ทั้งหมดให้เป็น Dark Mode */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(to right, #1e1e1e, #3a3a3a) !important;
            color: white;
        }
        
        /* ตั้งค่าสีตัวอักษรหลัก */
        [data-testid="stMarkdownContainer"] {
            color: white;
        }
        
        /* กล่องหลัก (Glassmorphism) */
        .main {
            background: rgba(30, 30, 30, 0.8);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0px 5px 20px rgba(0,0,0,0.3);
            backdrop-filter: blur(10px);
            margin-top: 30px;
        }

        /* ปุ่ม Predict - เพิ่ม Glow & 3D Hover */
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

        /* หัวข้อใหญ่ ๆ */
        h1 {
            text-align: center;
            font-size: 45px;
            font-weight: bold;
            background: linear-gradient(45deg, #ffcccb, #ff7777);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 2px 2px 10px rgba(255, 255, 255, 0.2);
            margin-bottom: 10px;
        }
        h3 {
            text-align: center;
            font-size: 22px;
            font-weight: 500;
            color: #ff9999;
            margin-top: 0px;
        }
        h2 {
            margin-top: 0px;
        }

        /* การ์ดแสดงผล - เพิ่ม Animation fade-in + hover scale */
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0px); }
        }
        .result-card {
            animation: fadeIn 0.8s ease-in-out forwards;
            background: rgba(50, 50, 50, 0.9);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 5px 20px rgba(255,255,255,0.1);
            text-align: center;
            transition: transform 0.3s, box-shadow 0.3s;
            backdrop-filter: blur(12px);
            margin-top: 20px;
        }
        .result-card:hover {
            transform: scale(1.05);
            box-shadow: 0px 5px 25px rgba(255,255,255,0.2);
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
        fixed_acidity = st.number_input("Fixed Acidity (ความเป็นกรดคงที่)", min_value=0.0, step=0.1)
        volatile_acidity = st.number_input("Volatile Acidity (ความเป็นกรดระเหย)", min_value=0.0, step=0.01)
        citric_acid = st.number_input("Citric Acid (กรดซิตริก)", min_value=0.0, step=0.01)
        residual_sugar = st.number_input("Residual Sugar (น้ำตาลตกค้าง)", min_value=0.0, step=0.1)
        chlorides = st.number_input("Chloride (คลอไรด์)", min_value=0.0, step=0.001)
        free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide (ซัลเฟอร์ไดออกไซด์ฟรี)", min_value=0.0, step=1.0)

    with col2:
        total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide (ซัลเฟอร์ไดออกไซด์ทั้งหมด)", min_value=0.0, step=1.0)
        density = st.number_input("Density (ความหนาแน่น)", min_value=0.0, step=0.0001)
        pH = st.number_input("pH (ค่า pH)", min_value=0.0, step=0.01)
        sulphates = st.number_input("Sulphates (ซัลเฟต)", min_value=0.0, step=0.01)
        alcohol = st.number_input("Alcohol (แอลกอฮอล์)", min_value=0.0, step=0.1)

# ------------------------- Predict Button -------------------------
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🔮 Predict Wine Quality"):
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
        <div class='result-card'>
            <h2 style='color: {result_color};'>{result_text}</h2>
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
        <a href="https://line.me/R/ti/p/@777xzgmw" target="_blank">💬 Chat กับ LINE Bot</a>
    </div>
    """,
    unsafe_allow_html=True
)
