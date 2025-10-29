import torchaudio
p = r"E:\new-voice\voice-checks\accent_app\test_audio\voice.wav"  # or your original path if earlier read succeeded
wave, sr = torchaudio.load(p)