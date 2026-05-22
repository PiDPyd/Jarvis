import openwakeword
from openwakeword.model import Model

import pyaudio
import wave
import numpy as np
import time
from faster_whisper import WhisperModel
import json

with open("config.json") as f:
    config = json.load(f)

RATE=config["audio"]["rate"]
CHUNK=config["audio"]["chunk"]
LANGUAGE = config["audio"]["language"]

model = Model(wakeword_models=["C:\\Users\\piyus\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\openwakeword\\resources\\models\\hey_jarvis_v0.1.onnx"],inference_framework="onnx")
microphone = pyaudio.PyAudio()
whisper_model = WhisperModel("tiny", device="cuda", compute_type="auto")

mic = microphone.open(rate=RATE, channels=1,format=pyaudio.paInt16,frames_per_buffer=CHUNK,input=True)

def audioloop():
    wakeword_mic = mic.read(1280, exception_on_overflow=False)
    wakefeed = np.frombuffer(wakeword_mic, dtype=np.int16)
    prediction = model.predict(wakefeed)
    if prediction["hey_jarvis_v0.1"] > 0.6:
        model.reset()
        print("jarvis is awakened. actively Listening...")
        SILENCE_DURATION = config["audio"]["silence_duration"]
        SILENCE_THRESHOLD = config["audio"]["silence_threshold"]
        frames = []
        silent_chunks = 0
        max_silent_chunks = int(RATE / CHUNK * SILENCE_DURATION)
        while True:
            jarvis_voice_input = mic.read(CHUNK, exception_on_overflow=False)
            frames.append(jarvis_voice_input)
            audio_chunk = np.frombuffer(jarvis_voice_input, dtype=np.int16)
            volume = np.abs(audio_chunk).mean()
            if volume < SILENCE_THRESHOLD:
                silent_chunks +=1
            else:
                silent_chunks = 0

            if silent_chunks >= max_silent_chunks:
                print("Silence detected.")
                break

        with wave.open("cache/command.wav", "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))

        t1 = time.time()
        segments, _ = whisper_model.transcribe("cache/command.wav",beam_size=3,language=LANGUAGE)
        command = " ".join([s.text for s in segments])
        t2 = time.time()
        print(f"Transcribe took: {t2-t1:.2f}s")
        return command




