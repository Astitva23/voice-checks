from speechbrain.pretrained import EncoderClassifier

print("🧠 Loading SpeechBrain model (VoxLingua107)...")
model = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa")
print("✅ Model loaded successfully.")

print("🎧 Identifying accent...")
result = model.classify_file("voice.wav")
print("Accent / Language Detected:", result)
