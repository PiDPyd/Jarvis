from modules import audio
from modules import tts

print("jarvis is online.")
while True:
    stt = audio.audioloop()
    if stt != None:
        tts.say(stt)