# train_accent_classifier.py
import os
import csv
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import torch

# Config
MODEL_NAME = "facebook/wav2vec2-base"
SAMPLE_RATE = 16000
EMBED_DIM = 768   # wav2vec2-base hidden size
CHUNK_SECONDS = 8  # chunk each file to ~8s windows for more samples

# Load feature extractor & model
print("Loading wav2vec2 model...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
wav2vec = Wav2Vec2Model.from_pretrained(MODEL_NAME)
wav2vec.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wav2vec.to(device)

def load_audio(path, sr=SAMPLE_RATE):
    y, orig_sr = sf.read(path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr, sr)
    return y

def get_embedding(wave_np):
    # wave_np: 1D numpy float32 audio at SAMPLE_RATE
    with torch.no_grad():
        inputs = processor(wave_np, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        outputs = wav2vec(input_values, output_hidden_states=False)
        hidden = outputs.last_hidden_state.squeeze(0)  # (T, D)
        emb = hidden.mean(dim=0).cpu().numpy()         # mean pooling -> (D,)
    return emb

# Read metadata.csv
meta_file = "metadata.csv"
if not os.path.exists(meta_file):
    raise SystemExit("Place metadata.csv (columns: filepath,label) in the script folder")

filepaths = []
labels = []
with open(meta_file, newline='', encoding='utf-8') as f:
    rdr = csv.reader(f)
    for row in rdr:
        if not row: continue
        path, lab = row[0].strip(), row[1].strip()
        if os.path.exists(path):
            filepaths.append(path)
            labels.append(lab)

print(f"Found {len(filepaths)} files in metadata.")

# Build dataset (extract embeddings; chunk long files)
X = []
y = []
for path, lab in tqdm(list(zip(filepaths, labels))):
    audio = load_audio(path)
    if len(audio) < SAMPLE_RATE*1:  # skip too short
        continue
    # chunking
    hop = SAMPLE_RATE * CHUNK_SECONDS
    start = 0
    while start < len(audio):
        chunk = audio[start:start+hop]
        if len(chunk)/SAMPLE_RATE >= 2.0:  # keep at least 2s
            emb = get_embedding(chunk.astype(np.float32))
            X.append(emb)
            y.append(lab)
        start += hop

X = np.vstack(X)
y = np.array(y)
print("Embeddings shape:", X.shape)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

# Scale + classifier
scaler = StandardScaler().fit(X_train)
Xtr = scaler.transform(X_train)
Xte = scaler.transform(X_test)

clf = LogisticRegression(max_iter=500, multi_class="multinomial", solver="saga")
print("Training classifier...")
clf.fit(Xtr, y_train)

# Eval
pred = clf.predict(Xte)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# Save artifacts
joblib.dump({"scaler": scaler, "clf": clf}, "accent_classifier.joblib")
print("Saved accent_classifier.joblib")
