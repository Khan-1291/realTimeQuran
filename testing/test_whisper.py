import whisper
import os
model =whisper.load_model("base")

print("model loaded")

if not os.path.exists("audio.mp3"):
    print("No audio.mp3 file exist")
else:    

    result= model.transcribe("audio.mp3",fp16=False)
    print("-" * 20)
    print(result)

    print(result["text"])