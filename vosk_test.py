from vosk import Model, KaldiRecognizer
import wave
import json

wf = wave.open("audioo.wav", "rb")

model = Model("vosk-model-ar-0.22-linto-1.1.0")

rec = KaldiRecognizer(model, wf.getframerate())

while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        print(rec.Result())

print(rec.FinalResult())