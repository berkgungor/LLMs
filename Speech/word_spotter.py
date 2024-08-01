import pvporcupine
import pyaudio
import struct
import whisper
import os
import numpy as np
from whisper_mic import WhisperMic
import re

# Initialize Porcupine
access_key = "RCz7vboZ9PFCx1PBXHbOtRWcsuUpH8eIZV6fh6SiARFjmYw0xZXr8w=="  # Replace with your Picovoice Access Key
keyword_paths = ["C:\\Users\\berkg\\source\\AI_Coach\\Speech\\Hey-Putt-View\\Hey-Putt-View_en_windows_v3_0_0.ppn"]
porcupine = pvporcupine.create(access_key=access_key, keyword_paths=keyword_paths)

mic = WhisperMic(model="medium", english=True, energy=300, pause=1, dynamic_energy=False, save_file=False, device="cuda", mic_index=None)

# Audio stream setup for Porcupine
pa = pyaudio.PyAudio()
audio_stream = pa.open(
    rate=porcupine.sample_rate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=porcupine.frame_length
)

print("Listening for wake word...")

while True:
    pcm = audio_stream.read(porcupine.frame_length)
    pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

    result = porcupine.process(pcm)
    if result >= 0:
        print("Wake word detected")
        result = mic.listen()
        if "parts" in result:
            result = result.replace("parts", "putts")
        if "pots" in result:
            result = result.replace("pots", "putts")
        print("Modified Transcription:", result)
        break
    

# Clean up
audio_stream.close()
pa.terminate()
porcupine.delete()