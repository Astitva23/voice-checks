# main_fixed.py
import os
import sys
import numpy as np
import soundfile as sf
import torchaudio
import torch

# --- choose classifier from SpeechBrain ---
from speechbrain.pretrained import EncoderClassifier

# -------------------------
# Utility: trim silence + normalize
# -------------------------
def trim_silence(y, sr, top_db=30):
    if y.ndim > 1:
        y = y.mean(axis=1)
    frame_len = int(0.03 * sr)
    hop = max(1, int(frame_len // 2))
    energy = []
    for i in range(0, max(1, len(y)-frame_len), hop):
        frame = y[i:i+frame_len]
        energy.append(np.sum(frame**2))
    if len(energy) == 0:
        return y
    energy = np.array(energy)
    thresh = np.max(energy) * 10**(-top_db / 10.0)
    frames = np.where(energy > thresh)[0]
    if len(frames) == 0:
        return y
    start = max(0, frames[0]*hop)
    end = min(len(y), frames[-1]*hop + frame_len)
    return y[start:end]

def normalize_rms(y, target_db=-20.0):
    rms = np.sqrt(np.mean(y**2)) if y.size else 0.0
    if rms == 0:
        return y
    target = 10**(target_db / 20.0)
    return y * (target / (rms + 1e-9))

# -------------------------
# Load model (with error handling)
# -------------------------
print("🧠 Loading SpeechBrain VoxLingua107 model (may download on first run)...")
try:
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-voxlingua107-ecapa",
        savedir="pretrained_models/lang-id-voxlingua107-ecapa"
    )
except Exception as e:
    print("\n❌ Failed to load SpeechBrain model. Error:")
    print(e)
    print("\nPossible fixes:")
    print(" - Ensure you have internet for the first run so model files can download.")
    print(" - If Windows raises symlink permission errors, run your terminal as Administrator OR set HF_HUB_DISABLE_SYMLINKS_WARNING=1")
    print(" - If the hub access fails with 401/403, your network or HF account may be blocking it.")
    sys.exit(1)

print("✅ Model loaded successfully.\n")

# -------------------------
# Input audio file (your recording)
# -------------------------
wav_in = "voice.wav"   # replace if your recorded file has another name
if not os.path.exists(wav_in):
    print(f"❗ Cannot find {wav_in}. Please record or place your audio file named '{wav_in}' next to this script.")
    sys.exit(1)

# load with soundfile for preprocessing (keeps dtype float32)
y, sr = sf.read(wav_in)
if y.ndim > 1:
    y = y.mean(axis=1)

# preprocess
y = trim_silence(y, sr, top_db=30)
y = normalize_rms(y, target_db=-20.0)

if len(y) / sr < 3.0:
    print("🔊 Recording too short after trimming. Try recording 8-12 seconds of continuous speech.")
    sys.exit(0)

# save cleaned temporary file
clean_path = "voice_clean.wav"
sf.write(clean_path, y, sr)

# load into torchaudio tensor
signal, fs = torchaudio.load(clean_path)   # shape (channels, samples)
# ensure mono
if signal.shape[0] > 1:
    signal = torch.mean(signal, dim=0, keepdim=True)

# resample to 16k if required
target_sr = 16000
if fs != target_sr:
    resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=target_sr)
    signal = resampler(signal)
    fs = target_sr

# Ensure tensor is float and on cpu
signal = signal.float()

# -------------------------
# Run classifier (file-based or batch)
# -------------------------
print("🎧 Running classification...")

try:
    # Using classify_file is simple and reliable
    result = classifier.classify_file(clean_path)
except Exception as e:
    # fallback: sometimes classify_file fails; try classify_batch with tensor
    try:
        print("classify_file failed, trying classify_batch(...)")
        out = classifier.classify_batch(signal)
        result = out
    except Exception as e2:
        print("❌ Classification failed. Errors:")
        print(e)
        print(e2)
        sys.exit(1)

# -------------------------
# Parse result (robust for tuple or other shapes) surefa
# -------------------------
labels = None
logits = None

if isinstance(result, tuple) or isinstance(result, list):
    # common SpeechBrain returns: (logits_tensor, _, _, labels_list)
    if len(result) >= 4:
        logits = result[0].squeeze()
        labels = result[3]
    else:
        # attempt to infer
        # e.g., some variants return (logits,)
        logits = result[0].squeeze() if len(result) >= 1 else None
else:
    # unknown format: print raw result and exit
    print("Raw result (unknown format):", result)
    sys.exit(0)

if logits is None or labels is None:
    print("❌ Could not parse model output. Raw result printed above.")
    sys.exit(1)

probs = torch.softmax(logits, dim=0)
topk = torch.topk(probs, k=min(5, probs.numel()))

print("\n🔊 Top 5 predictions (language/variety — probability):")
for idx, p in zip(topk.indices.tolist(), topk.values.tolist()):
    label = labels[idx] if idx < len(labels) else f"label_{idx}"
    print(f"  {label} — {p*100:.2f}%")

# -------------------------
# Filter for English-related labels (heuristic)
# -------------------------
english_keywords = ["en", "english", "british", "american", "us", "uk", "australian", "india", "indian"]
eng_candidates = []
for i in range(len(labels)):
    lbl = labels[i].lower()
    if any(k in lbl for k in english_keywords):
        eng_candidates.append((labels[i], probs[i].item()))

if eng_candidates:
    eng_candidates_sorted = sorted(eng_candidates, key=lambda x: x[1], reverse=True)
    print("\n🔎 English-related candidates (heuristic filter):")
    for lbl, p in eng_candidates_sorted[:5]:
        print(f"  {lbl} — {p*100:.2f}%")
else:
    print("\n🔎 No English-related labels appeared in top candidates. (Model may be detecting speaker L1 at higher score.)")

print("\n✅ Done.")
