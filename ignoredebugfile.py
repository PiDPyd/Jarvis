import openwakeword
from openwakeword.model import Model

import pyaudio
import wave
import numpy as np
import time


model = Model(wakeword_models=["C:\\Users\\piyus\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\openwakeword\\resources\\models\\hey_jarvis_v0.1.onnx"],inference_framework="onnx")
microphone = pyaudio.PyAudio()

audiofeed = microphone.open(rate=16000, channels=1,format=pyaudio.paInt16,frames_per_buffer=1280,input=True)

print("jarvis is online")

while True:
    rawfeed = audiofeed.read(1280, exception_on_overflow=False)
    wakefeed = np.frombuffer(rawfeed, dtype=np.int16)
    prediction = model.predict(wakefeed)
    print(prediction)
    if prediction["hey_jarvis_v0.1"] > 0.7:
        model.reset() 
        print("pretend this is listening")





