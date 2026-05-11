import openwakeword
from openwakeword.model import Model

import pyaudio
import numpy as np

model = Model(wakeword_models=["C:\\Users\\piyus\\AppData\\Local\\Programs\\Python\\Python38\\lib\\site-packages\\openwakeword\\resources/models/hey_jarvis_v0.1.tflite"])
microphone = pyaudio.PyAudio()

audiofeed = microphone.open(rate=16000, channels=1,format=pyaudio.paInt16,frames_per_buffer=1280,input=True)
print("Jarvis is listening...")
while True:
    rawfeed = audiofeed.read(1280, exception_on_overflow=False)
    wakefeed = np.frombuffer(rawfeed, dtype=np.int16)
    prediction = model.predict(wakefeed)
    if prediction["hey_jarvis_v0.1"] > 0.7:
        print("Detected.")



