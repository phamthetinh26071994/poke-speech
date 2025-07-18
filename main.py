from flask import Flask, request, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel
import os
import uuid

app = Flask(__name__)
CORS(app)  # ✅ Cho phép mọi origin truy cập

# ✅ Dùng model 'tiny' để tiết kiệm RAM trên Replit
model = WhisperModel("tiny", device="cpu", compute_type="int8")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "Không có file ghi âm"}), 400

    file = request.files["audio"]

    # ✅ Tạo tên file tạm với đuôi gốc
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".webm", ".wav", ".mp3", ".m4a", ".ogg"]:
        return jsonify({"error": f"Định dạng không hỗ trợ: {ext}"}), 400

    temp_name = f"temp_{uuid.uuid4().hex}{ext}"
    file.save(temp_name)

    try:
        segments, info = model.transcribe(temp_name)
        transcript = " ".join([seg.text for seg in segments])
        return jsonify({ "transcript": transcript })
    except Exception as e:
        return jsonify({ "error": str(e) }), 500
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)

# ✅ Replit sẽ tự chạy app nếu có file main.py
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
