from flask import Flask, request, jsonify
import whisper
import os

app = Flask(__name__)

# ✅ Load model Whisper (Trainer có thể dùng 'tiny' nếu muốn nhẹ hơn)
model = whisper.load_model("base")  # hoặc 'tiny' nếu cần tiết kiệm tài nguyên

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "Không có file ghi âm"}), 400

    file = request.files["audio"]
    filepath = "temp.wav"
    file.save(filepath)

    try:
        result = model.transcribe(filepath)
        transcript = result["text"]
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(filepath)

    return jsonify({"transcript": transcript})

# ✅ Chạy app trên port 8080 để Render nhận đúng
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
