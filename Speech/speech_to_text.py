from whisper_mic import WhisperMic

mic = WhisperMic(model="medium", english=True, energy=300, pause=1, dynamic_energy=False, save_file=False, device="cpu", mic_index=None)
result = mic.listen_loop()
print(result)