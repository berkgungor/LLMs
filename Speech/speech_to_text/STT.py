import io
import os
import speech_recognition as sr
import torch
import nltk
from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform
from faster_whisper import WhisperModel
#from translatepy.translators.google import GoogleTranslate
#from TranscriptionWindow import TranscriptionWindow
from faster_whisper.tokenizer import _LANGUAGE_CODES, Tokenizer
import whisper

class FasterSpeech:
    def __init__(self):
        self.phrase_time = None
        self.last_sample = bytes()
        self.data_queue = Queue()
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = 2000
        self.recorder.dynamic_energy_threshold = False

        self.speechLangauge = "English" # German
        self.model = "medium" # medium, medium.en, large-v1, large-v2
        self.device = "auto" #gpu
        self.compute_type = "int8"
        self.cpu_threads = 0
        self.record_timeout = 2
        self.phrase_timeout = 3
        self.default_microphone = 'pulse'
        #nltk.download('punkt')
        self.transcription = ['']
        
        if self.speechLangauge == "English":
            self.model = self.model + ".en"

        if 'linux' in platform:
            mic_name = self.default_microphone
            if not mic_name or mic_name == 'list':
                print("Available microphone devices are: ")
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    print(f"Microphone with name \"{name}\" found")   
                return
            else:
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    if mic_name in name:
                        self.source = sr.Microphone(sample_rate=16000, device_index=index)
                        break
        else:
            self.source = sr.Microphone(sample_rate=16000)
        
        self.audio_model = WhisperModel(self.model, device=self.device, compute_type=self.compute_type, cpu_threads=self.cpu_threads)
        
    def adjust_for_ambient_noise(self):
        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)
            
        self.listen_in_background()
            
    def record_callback(self,_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        self.data_queue.put(data)
        
    def listen_in_background(self):
        self.recorder.listen_in_background(self.source, self.record_callback, phrase_time_limit=self.record_timeout)

    def process_audio_data(self):
        last_data_time = None
        while True:
            try:
                now = datetime.utcnow()
                if not self.data_queue.empty():
                    phrase_complete = False
                    if self.phrase_time and now - self.phrase_time > timedelta(seconds=self.phrase_timeout):
                        self.last_sample = bytes()
                        phrase_complete = True
                    self.phrase_time = now

                    while not self.data_queue.empty():
                        data = self.data_queue.get()
                        self.last_sample += data

                    audio_data = sr.AudioData(self.last_sample, self.source.SAMPLE_RATE, self.source.SAMPLE_WIDTH)
                    wav_data = io.BytesIO(audio_data.get_wav_data())

                    with open(self.temp_file, 'w+b') as f:
                        f.write(wav_data.read())

                    text = ""
                    if self.speechLangauge == "English":
                        segments, info = self.audio_model.transcribe(self.temp_file,language="en")
                    elif self.speechLangauge == "German":
                        segments, info = self.audio_model.transcribe(self.temp_file,language="de")
                        
                    for segment in segments:
                        text += segment.text

                    if phrase_complete:
                        self.transcription.append(text)
                    else:
                        self.transcription[-1] = text
                    sleep(0.25)
                    last_data_time = now
                    
                elif last_data_time and now - last_data_time > timedelta(seconds=2):
                    break
            except KeyboardInterrupt:
                break

        print("\n\nTranscription:")
        #return transcription
        return self.transcription
    
    def modify_transcription(self,raw_transcription):
        if "sample" in raw_transcription:
            raw_transcription = raw_transcription.replace("sample", "sample")
            return raw_transcription
    
    def test_audio(self):
        model = whisper.load_model("base")
        result = model.transcribe("/Users/berkgungor/Downloads/Anamnese2people.m4a")
        print(f' The text : \n {result["text"]}')
        

    def run_transcriber(self):
        self.temp_file = NamedTemporaryFile().name 
        self.adjust_for_ambient_noise()
        print("Ready to go...")
        output = self.process_audio_data()
        sentence = ' '.join(output)
        #clean_transcription = self.modify_transcription(sentence)
        return sentence
       
transcriber  = FasterSpeech()
test = transcriber.test_audio()
#transcription = transcriber.run_transcriber()
#print(transcription)

