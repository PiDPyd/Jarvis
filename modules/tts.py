from kokoro import KPipeline
import sounddevice
import json

with open("config.json") as f:
    config = json.load(f)

VOICE = config["tts"]["voice_model"]
SPEED = config["tts"]["speed"]

pipeline = KPipeline(lang_code="a")

def say(text):
    generator = pipeline(text, voice=VOICE, speed=SPEED)
    for _,_, audio in generator:
        sounddevice.play(audio, samplerate=24000)
        sounddevice.wait()