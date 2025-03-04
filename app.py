from flask import Flask, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import numpy as np
import joblib
import os

app = Flask(__name__)

# ตรวจสอบว่า Environment Variables ถูกตั้งค่าหรือไม่
LINE_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

if not LINE_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    raise ValueError("❌ Missing LINE_ACCESS_TOKEN or LINE_CHANNEL_SECRET. Please set them in Railway Environment Variables.")

line_bot_api = LineBotApi(LINE_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# โหลดโมเดลและ Scaler
try:
    model = joblib.load("best_wine_quality_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    raise ValueError(f"❌ Error loading model or scaler: {e}")

@app.route("/")
def home():
    return "🚀 LINE Chatbot & Wine Quality API is Running!"

@app.route("/webhook", methods=["POST"])
def webhook():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)
    
    if not signature:
        return jsonify({"status": "error", "message": "Missing signature"}), 400

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        return jsonify({"status": "error", "message": "Invalid signature"}), 400
    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    try:
        text = event.message.text.strip()
        values = list(map(float, text.split(",")))

        if len(values) != 11:
            reply_text = "กรุณากรอกข้อมูล 11 ค่า เช่น: 7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4"
        else:
            input_data = np.array([values])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            quality_score = int(probability[1] * 10)

            if prediction == 1:
                reply_text = f"🍷 คุณภาพไวน์: สูง (ระดับ {quality_score}/10)"
            else:
                reply_text = f"⚠️ คุณภาพไวน์: ต่ำ (ระดับ {10 - quality_score}/10)"

        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

    except Exception as e:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="เกิดข้อผิดพลาด โปรดลองใหม่\n" + str(e)))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Railway จะกำหนด PORT ให้อัตโนมัติ
    app.run(host="0.0.0.0", port=port, debug=True)
