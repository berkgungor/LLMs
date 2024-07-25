import pyttsx3
from elevenlabs import Voice, VoiceSettings, play, stream
from elevenlabs.client import ElevenLabs

class STT():
    def __init__(self):
        self.engine = pyttsx3.init()
        self.voices = self.engine.getProperty('voices')
        self.elevenlabs_api_key = "9f10c4238c3092caaa3f6120fc06549a"

    def elevenlabs(self,input_text):
        self.client = ElevenLabs(api_key=self.elevenlabs_api_key)
        #response = self.client.voices.get_all()
        audio = self.client.generate(text=input_text, voice="Brian" ,model="eleven_multilingual_v1")
        return audio


#stt = STT()
#audio = stt.elevenlabs("hello there, can we practice some putt together?")
#play(audio)