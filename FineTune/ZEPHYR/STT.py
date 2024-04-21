import pvporcupine
import pyaudio
import struct
import whisper
from whisper_mic import WhisperMic
import numpy as np
import re

class VoiceAssistant:
    def __init__(self, access_key = "RCz7vboZ9PFCx1PBXHbOtRWcsuUpH8eIZV6fh6SiARFjmYw0xZXr8w==", keyword_paths = ["C:\\Users\\berkg\\source\\AI_Coach\\Speech\\Hey-Putt-View\\Hey-Putt-View_en_windows_v3_0_0.ppn"], model='medium.en'):
        # Initialize Porcupine for wake word detection
        self.porcupine = pvporcupine.create(access_key=access_key, keyword_paths=keyword_paths)
        self.mic = WhisperMic(model="medium", english=True, energy=300, pause=1, dynamic_energy=False, save_file=False, device="cuda", mic_index=None)
        self.pa = pyaudio.PyAudio()
        self.audio_stream = self.pa.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length
        )
        
    def listen_and_respond(self):
        print("Listening for wake word...")
        while True:
            pcm = self.audio_stream.read(self.porcupine.frame_length)
            pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)

            result = self.porcupine.process(pcm)
            if result >= 0:
                print("Wake word detected")
                result = self.mic.listen()
                if "parts" in result:
                    result = result.replace("parts", "putts")
                if "pots" in result:
                    result = result.replace("pots", "putts")
                if "pot" in result:
                    result = result.replace("pot", "putt")
                if "spit" in result:
                    result = result.replace("spit", "speed")
                return result

    def cleanup(self):
        self.audio_stream.close()
        self.pa.terminate()
        self.porcupine.delete()


# Main execution
if __name__ == "__main__":
    access_key = "RCz7vboZ9PFCx1PBXHbOtRWcsuUpH8eIZV6fh6SiARFjmYw0xZXr8w=="  # Replace with your Picovoice Access Key
    keyword_paths = ["C:\\Users\\berkg\\source\\AI_Coach\\Speech\\Hey-Putt-View\\Hey-Putt-View_en_windows_v3_0_0.ppn"]
    assistant = VoiceAssistant(access_key, keyword_paths)
    text = assistant.listen_and_respond()
    assistant.cleanup()
