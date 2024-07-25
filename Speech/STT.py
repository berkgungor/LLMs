import pvporcupine
import pyaudio
import struct
import whisper
from whisper_mic import WhisperMic
import numpy as np
import re

class VoiceAssistant:
    def __init__(self, access_key, keyword_paths, model='medium.en'):
        # Initialize Porcupine for wake word detection
        self.porcupine = pvporcupine.create(access_key=access_key, keyword_paths=keyword_paths)
        self.pa = pyaudio.PyAudio()
        self.audio_stream = self.pa.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length
        )

        # Initialize Whisper for speech-to-text
        self.engine = whisper.load_model(model)
        self.mic = WhisperMic(model="medium", english=True, energy=300, pause=1, dynamic_energy=False, save_file=False, device="cuda", mic_index=None)

    def record_audio(self, duration):
        print("Recording...")
        frames = []
        for _ in range(0, int(self.porcupine.sample_rate / 1024 * duration)):
            data = self.audio_stream.read(1024)
            frames.append(np.frombuffer(data, dtype=np.int16))

        return np.concatenate(frames)

    def speech_to_text(self, audio_data):
        result = self.engine.transcribe(audio_data, sample_rate=self.porcupine.sample_rate)
        return result["text"]

    def listen_and_respond(self):
        print("Listening for wake word...")
        while True:
            pcm = self.audio_stream.read(self.porcupine.frame_length)
            pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)

            result = self.porcupine.process(pcm)
            if result >= 0:
                print("Wake word detected")
                audio_data = self.record_audio(5)  # Record for 5 seconds
                transcription = self.speech_to_text(audio_data)

                # Post-processing transcription as per WhisperMic functionality
                transcription = re.sub(r"\bparts\b", "putts", transcription)
                transcription = re.sub(r"\bparts\b", "putts", transcription)

                print("Modified Transcription:", transcription)
                break

    def cleanup(self):
        self.audio_stream.close()
        self.pa.terminate()
        self.porcupine.delete()


# Main execution
if __name__ == "__main__":
    access_key = "RCz7vboZ9PFCx1PBXHbOtRWcsuUpH8eIZV6fh6SiARFjmYw0xZXr8w=="  # Replace with your Picovoice Access Key
    keyword_paths = ["C:\\Users\\berkg\\source\\AI_Coach\\Speech\\Hey-Putt-View\\Hey-Putt-View_en_windows_v3_0_0.ppn"]
    assistant = VoiceAssistant(access_key, keyword_paths)
    assistant.listen_and_respond()
    assistant.cleanup()
