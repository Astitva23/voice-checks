import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os

# ✅ Use stable model (public, large, multilingual)
MODEL_ID = "facebook/wav2vec2-large-xlsr-53"

print("🔁 Loading pretrained model and feature extractor...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
model = Wav2Vec2Model.from_pretrained(MODEL_ID)
model.eval()

def extract_features(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    inputs = feature_extractor(
        waveform.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

# ✅ Training dataset (add your labelled wav files here)
dataset = {
    "indian": [
        "samples/indian_1.wav",
        "samples/indian_2.wav"
    ],
    "american": [
        "samples/american_1.wav",
        "samples/american_2.wav"
    ]
}

print("🎙️ Extracting features from training dataset...")
X, y = [], []
for label, files in dataset.items():
    for f in files:
        if os.path.exists(f):
            feat = extract_features(f)
            X.append(feat)
            y.append(label)

if len(X) == 0:
    raise RuntimeError("No training audio found in 'samples' directory!")

X, y = np.array(X), np.array(y)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
print("✅ Model trained on", len(X), "samples")

# ✅ Prediction
test_file = "voice.wav"  # your input file
if not os.path.exists(test_file):
    raise FileNotFoundError("Missing test file voice.wav")

print("🎧 Predicting accent...")
test_feat = extract_features(test_file)
pred = knn.predict([test_feat])[0]
print(f"🎤 Detected accent: {pred.upper()}")
