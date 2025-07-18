from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import os

app = Flask(__name__)

# ✅ Dùng model 'tiny' để chạy được trên Render
model = WhisperModel("tiny", device="cpu", compute_type="int8")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "Không có file ghi âm"}), 400

    file = request.files["audio"]
    filepath = "temp.wav"
    file.save(filepath)

    try:
        segments, info = model.transcribe(filepath)
        transcript = " ".join([seg.text for seg in segments])
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(filepath)

    return jsonify({"transcript": transcript})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
