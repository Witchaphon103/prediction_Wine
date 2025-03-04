import os

# ดึงค่า Access Token & Secret จาก Environment Variables
LINE_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

# ✅ เพิ่มโค้ดนี้เพื่อตรวจสอบค่าใน Railway Logs
print(f"🔍 LINE_ACCESS_TOKEN: {LINE_ACCESS_TOKEN}")
print(f"🔍 LINE_CHANNEL_SECRET: {LINE_CHANNEL_SECRET}")

# ถ้าค่าเป็น None ให้แจ้งเตือน
if not LINE_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    raise ValueError("❌ Missing LINE_ACCESS_TOKEN or LINE_CHANNEL_SECRET. Check Railway Environment Variables.")
