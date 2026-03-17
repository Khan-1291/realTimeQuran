import whisper
from quran_loader import load_quran

# Load Quran text
quran = load_quran()

audio_file = "datasets/everyayah/001001.mp3"

print("Loading Whisper model...")
model = whisper.load_model("base")
print("Model loaded successfully")

print("Transcribing audio... please wait")

# Speech to text
result = model.transcribe(audio_file)

predicted = result["text"]
expected = quran["001001"]

print("*" * 30)
print("Predicted:", predicted)
print("Expected :", expected)