"""
accent_knn_wav2vec.py

Usage:
  1) Put your test file as voice.wav next to this script (or change INPUT_PATH).
  2) Prepare prototypes in prototypes/<accent>/*.wav (see README above).
  3) Run: python accent_knn_wav2vec.py

Outputs:
  - per-prototype cosine similarities
  - per-accent averaged similarity (ranked)
  - saves/loads prototype embeddings in prototypes/.embeddings.npy for speed
"""

import os
import glob
import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.preprocessing import normalize
from collections import defaultdict

# ---------------- CONFIG ----------------
INPUT_PATH = "voice.wav"         # test file
PROTO_DIR = "prototypes"         # prototypes root
MODEL_NAME = "facebook/wav2vec2-base"  # embedding model
SAMPLE_RATE = 16000
EMB_POOL = "mean"   # mean pooling over time; other options could be max
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_EMBS = True    # cache prototype embeddings for faster runs
EMB_CACHE_FILE = os.path.join(PROTO_DIR, ".embeddings.npy")
META_CACHE_FILE = os.path.join(PROTO_DIR, ".meta.npy")
# similarity thresholds (tune per your data)
STRONG_SIM = 0.65
MEDIUM_SIM = 0.55

# ---------------- Helpers ----------------
def load_audio_mono(path, sr=SAMPLE_RATE):
    y, orig_sr = sf.read(path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if orig_sr != sr:
        y = librosa.resample(y.astype(np.float32), orig_sr, sr)
    return y.astype(np.float32)

def get_embedding(wave_np, processor, model):
    # wave_np: 1D numpy float32 at SAMPLE_RATE
    inputs = processor(wave_np, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(DEVICE)
    with torch.no_grad():
        outputs = model(input_values)
        hidden = outputs.last_hidden_state.squeeze(0).cpu().numpy()  # (T, D)
    if EMB_POOL == "mean":
        emb = np.mean(hidden, axis=0)
    elif EMB_POOL == "max":
        emb = np.max(hidden, axis=0)
    else:
        emb = np.mean(hidden, axis=0)
    return emb / (np.linalg.norm(emb) + 1e-9)

# ---------------- Load model ----------------
print("Loading wav2vec2 model (may take a few seconds)...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print("Model ready.\n")

# ---------------- Load or compute prototype embeddings ----------------
def build_prototype_embeddings(proto_root=PROTO_DIR, rebuild=False):
    protos = []      # list of (accent_label, filepath)
    accents = []
    for accent_dir in sorted(glob.glob(os.path.join(proto_root, "*"))):
        if not os.path.isdir(accent_dir):
            continue
        accent = os.path.basename(accent_dir)
        files = sorted(glob.glob(os.path.join(accent_dir, "*.wav")))
        if not files:
            continue
        for f in files:
            protos.append((accent, f))
        if accent not in accents:
            accents.append(accent)

    if len(protos) == 0:
        print("No prototypes found in", proto_root)
        return None, None, None

    # if cache exists and not rebuild -> load
    if os.path.exists(EMB_CACHE_FILE) and os.path.exists(META_CACHE_FILE) and not rebuild:
        try:
            meta = np.load(META_CACHE_FILE, allow_pickle=True).item()
            emb_matrix = np.load(EMB_CACHE_FILE)
            print(f"Loaded cached {emb_matrix.shape[0]} prototype embeddings.")
            return emb_matrix, meta, accents
        except Exception:
            pass

    # compute embeddings
    emb_list = []
    meta = []  # list of dicts: {accent, filepath}
    print("Computing prototype embeddings for", len(protos), "files...")
    for accent, fp in protos:
        wav = load_audio_mono(fp)
        emb = get_embedding(wav, processor, model)
        emb_list.append(emb)
        meta.append({"accent": accent, "file": fp})
    emb_matrix = np.vstack(emb_list)
    # normalize rows (should already be unit norm)
    emb_matrix = normalize(emb_matrix, axis=1)
    if SAVE_EMBS:
        os.makedirs(proto_root, exist_ok=True)
        np.save(EMB_CACHE_FILE, emb_matrix)
        np.save(META_CACHE_FILE, np.array(meta, dtype=object))
        print("Saved prototype embeddings cache.")
    return emb_matrix, meta, accents

emb_matrix, meta, accents = build_prototype_embeddings(PROTO_DIR, rebuild=False)
if emb_matrix is None:
    print("No prototype embeddings. Create prototypes in 'prototypes/<accent>/*.wav' and rerun.")
    # optionally, we could offer to record prototypes here — but we'll stop
    raise SystemExit

# Build per-accent indices
accent_to_idxs = defaultdict(list)
for i, m in enumerate(meta):
    accent_to_idxs[m["accent"]].append(i)

# ---------------- Process input file ----------------
if not os.path.exists(INPUT_PATH):
    print(f"Input file '{INPUT_PATH}' not found. Record a file named '{INPUT_PATH}' or change INPUT_PATH.")
    raise SystemExit

wav = load_audio_mono(INPUT_PATH)
if len(wav) / SAMPLE_RATE < 5.0:
    print("Input is short (<5s). Try to provide ~10-15s reading of sentences for best results.")

query_emb = get_embedding(wav, processor, model)
query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-9)  # unit norm

# ---------------- Similarities ----------------
# cosine similarity = dot product since normalized
sims = emb_matrix.dot(query_emb)   # shape (num_protos,)
# per-prototype display
print("\nPer-prototype similarities (top 10):")
proto_list = sorted([(i, sims[i], meta[i]["accent"], meta[i]["file"]) for i in range(len(sims))], key=lambda x: x[1], reverse=True)
for i, sim, acc, fp in proto_list[:10]:
    print(f"  {acc} | {os.path.basename(fp)} -> {sim:.4f}")

# per-accent average similarity (mean of prototypes)
accent_scores = {}
for acc in accents:
    idxs = accent_to_idxs[acc]
    vals = sims[idxs]
    # you can use mean or median; median is robust to outliers
    accent_scores[acc] = float(np.median(vals))

# sort and print
sorted_acc = sorted(accent_scores.items(), key=lambda x: x[1], reverse=True)
print("\nPer-accent median similarity (ranked):")
for acc, score in sorted_acc:
    print(f"  {acc} -> {score:.4f}")

best_acc, best_score = sorted_acc[0]
# Friendly interpretation and thresholds
if best_score >= STRONG_SIM:
    strength = "strong"
elif best_score >= MEDIUM_SIM:
    strength = "moderate"
else:
    strength = "weak"

print(f"\nFinal guess: {best_acc} (similarity {best_score:.3f}) — {strength} match")

# If you want, show per-accent percentages normalized (for display only)
norm_total = sum(max(0.0, v) for _, v in sorted_acc) or 1e-9
print("\nDisplay-style percentages (not probabilistic):")
for acc, score in sorted_acc:
    pct = (max(0.0, score) / norm_total) * 100.0
    print(f"  {acc}: {pct:.1f}%")

# Save query embedding (optional)
np.save("last_query_embedding.npy", query_emb)
print("\nDone.")
