from flask import Flask, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import numpy as np
import joblib

app = Flask(__name__)


LINE_ACCESS_TOKEN = "nTuc8iPZAO7CYX0xqDnpep1ZIBmdqKDcH8tI2pP7GWgKdC/B3PIVJ+/HDNJ69F48yniCMP+HsRJu0SGX4JyW1rKehbhwDmbuRd7F8yef1qj567O2fti9RhYVXleEGtMwcNNiZe6OVUv64y0sBmgxbQdB04t89/1O/w1cDnyilFU="
LINE_CHANNEL_SECRET = "ccc79951dcf9c6480fa521e45a902cae"
line_bot_api = LineBotApi(LINE_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)


model = joblib.load("best_wine_quality_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/webhook", methods=["POST"])
def webhook():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)
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
            reply_text = "กรุณากรอกข้อมูล 11 ค่า โดยใช้เครื่องหมายจุลภาค (,) คั่น เช่น: 7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4"
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

        
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply_text)
        )

    except Exception as e:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="เกิดข้อผิดพลาด โปรดลองใหม่\n"+str(e))
        )

if __name__ == "__main__":
    app.run(port=5000, debug=True)
