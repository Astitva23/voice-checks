# english_accent_integrated.py
import os
import sys
import time
import numpy as np
import soundfile as sf
import torchaudio
import torch
import sounddevice as sd
import wavio
import librosa

# Prevent HF from creating symlinks on Windows (do this before any HF usage)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

from speechbrain.pretrained import EncoderClassifier
from collections import defaultdict

# -------------------------
# CONFIG (tweak if needed)
# -------------------------
# If 'voice.wav' exists in the folder it will be used.
# If not, the script will record audio (see RECORDING settings).
wav_in = "voice.wav"            # default input filename
clean_path = "voice_clean.wav"
chunk_seconds = 12              # longer chunks for more stable prediction
hop_seconds = 3                 # overlap
target_sr = 16000
min_speech_seconds = 10.0       # require at least this much speech after trimming
record_if_missing = True        # if wav_in missing, record automatically
record_seconds_when_missing = 10

# -------------------------
# RECORDING (optional)
# -------------------------
def record_to_file(filename, duration_s=12, fs=16000, device=None):
    print(f"🔴 Recording for {duration_s} seconds... speak now.")
    recording = sd.rec(int(duration_s * fs), samplerate=fs, channels=1, dtype='int16', device=device)
    sd.wait()
    wavio.write(filename, recording, fs, sampwidth=2)
    print(f"✅ Saved recording to {filename}")

# -------------------------
# Preprocessing utilities
# -------------------------
def trim_silence(y, sr, top_db=30):
    if y.ndim > 1:
        y = y.mean(axis=1)
    frame_len = int(0.03 * sr)
    hop = max(1, frame_len // 2)
    energy = []
    for i in range(0, max(1, len(y)-frame_len), hop):
        f = y[i:i+frame_len]
        energy.append(np.sum(f**2))
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

# small music detector to skip highly musical chunks (optional)
def is_musical(y_np, sr):
    if len(y_np) < sr:  # less than 1s -> not musical
        return False
    # compute stft magnitude
    S = np.abs(librosa.stft(y_np.astype(np.float32), n_fft=1024, hop_length=512))
    spec_cent = np.mean(librosa.feature.spectral_centroid(S=S, sr=sr))
    # spectral flux approximate
    flux = np.mean(np.sum(np.diff(S, axis=1)**2, axis=0))
    # heuristics (tune if needed)
    if spec_cent > 3000 and flux > 1e6:
        return True
    return False

# -------------------------
# Load VoxLingua model
# -------------------------
print("🧠 Loading VoxLingua107 model (SpeechBrain). This may download files on first run...")
try:
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-voxlingua107-ecapa",
        savedir="pretrained_models/lang-id-voxlingua107-ecapa"
    )
except Exception as e:
    print("\n❌ Failed to load model. Error:")
    print(e)
    print("\nTry: (1) ensure internet; (2) run terminal as Administrator once; (3) set HF_HUB_DISABLE_SYMLINKS=1 env var.")
    sys.exit(1)
print("✅ Model loaded.\n")

# -------------------------
# Ensure input file exists (or record)
# -------------------------
if not os.path.exists(wav_in):
    if record_if_missing:
        try:
            record_to_file(wav_in, duration_s=record_seconds_when_missing, fs=target_sr)
        except Exception as e:
            print("❌ Recording failed:", e)
            sys.exit(1)
    else:
        print(f"❗ File '{wav_in}' not found. Please record or place audio named '{wav_in}' here.")
        sys.exit(1)

# -------------------------
# Load + preprocess audio
# -------------------------
y, sr = sf.read(wav_in)   # float32
if y.ndim > 1:
    y = y.mean(axis=1)

y = trim_silence(y, sr, top_db=30)
y = normalize_rms(y, target_db=-20.0)

if len(y)/sr < min_speech_seconds:
    print(f"🔊 Audio too short after trimming ({len(y)/sr:.2f}s). Please record 8-15s of continuous speech.")
    sys.exit(0)

# write cleaned wav
sf.write(clean_path, y, sr)

# load into torch tensor
signal, fs = torchaudio.load(clean_path)   # (channels, samples)
if signal.shape[0] > 1:
    signal = torch.mean(signal, dim=0, keepdim=True)
# resample to target_sr if required
if fs != target_sr:
    resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=target_sr)
    signal = resampler(signal)
    fs = target_sr

signal = signal.float()
total_samples = signal.shape[1]
samples_per_chunk = int(chunk_seconds * fs)
hop = int(hop_seconds * fs)

# create chunk list
chunks = []
starts = list(range(0, total_samples, hop))
for s in starts:
    e = min(total_samples, s + samples_per_chunk)
    chunk = signal[:, s:e]
    # keep chunk with at least 1s
    if chunk.shape[1] >= int(1.0 * fs):
        if chunk.shape[1] < samples_per_chunk:
            pad = torch.zeros((1, samples_per_chunk - chunk.shape[1]))
            chunk = torch.cat([chunk, pad], dim=1)
        chunks.append((s, chunk))

if len(chunks) == 0:
    print("❗ No valid chunks extracted. Exiting.")
    sys.exit(1)

# ---------- Ensemble: file-level + chunk-level aggregation ----------
import math

print("🔁 Running file-level classification (stable for whole audio)...")
file_result = None
try:
    file_result = classifier.classify_file(clean_path)
except Exception as e:
    print("⚠️ classify_file failed:", e)

file_probs = {}
if isinstance(file_result, (list, tuple)) and len(file_result) >= 4:
    file_logits = file_result[0].squeeze()
    file_labels = list(file_result[3])
    try:
        fp = torch.softmax(file_logits, dim=0).cpu().numpy()
        for lbl, p in zip(file_labels[: len(fp)], fp):
            file_probs[lbl] = float(p)
    except Exception as e:
        print("⚠️ Could not convert file logits -> probs:", e)

# We still run chunk aggregation (label-name based) like before
print(f"\n🔊 Running chunk aggregation for {len(chunks)} chunks...")
label_prob_sums = defaultdict(float)
successful_chunks = 0
failed_chunks = 0

for idx, (start_idx, chunk) in enumerate(chunks):
    chunk_np = chunk.squeeze(0).cpu().numpy()
    try:
        if is_musical(chunk_np, fs):
            # skip musical chunks
            # print(f"  (skipping chunk {idx} — musical)")
            failed_chunks += 1
            continue
    except Exception:
        pass

    try:
        out = classifier.classify_batch(chunk)
    except Exception:
        try:
            out = classifier.classify_file(clean_path)
        except Exception as e:
            failed_chunks += 1
            continue

    if isinstance(out, (list, tuple)) and len(out) >= 4:
        logits = out[0].squeeze()
        labels = list(out[3])
    else:
        failed_chunks += 1
        continue

    try:
        probs = torch.softmax(logits, dim=0).cpu().numpy()
    except Exception:
        failed_chunks += 1
        continue

    L = min(len(labels), probs.shape[0])
    labels = labels[:L]
    probs = probs[:L]

    for lbl, p in zip(labels, probs):
        label_prob_sums[lbl] += float(p)

    successful_chunks += 1

if successful_chunks == 0 and not file_probs:
    print("❌ Both chunk aggregation and file-level classification failed. Exiting.")
    sys.exit(1)

# Mean chunk-level probs (divide sums by successful_chunks)
chunk_mean_probs = {}
if successful_chunks > 0:
    for lbl, s in label_prob_sums.items():
        chunk_mean_probs[lbl] = s / successful_chunks

# Build unified vocabulary of labels (union of file and chunk labels)
all_labels = set(list(file_probs.keys()) + list(chunk_mean_probs.keys()))

# Combine file-level and chunk-level with weights
# use stronger file-level weight
file_weight = 0.90
chunk_weight = 0.10


combined_probs = {}
for lbl in all_labels:
    f = file_probs.get(lbl, 0.0)
    c = chunk_mean_probs.get(lbl, 0.0)
    combined = file_weight * f + chunk_weight * c
    combined_probs[lbl] = combined

# -------------------------
# SAFER NORMALIZATION & ENGLISH-BUCKET MAPPING (replacement)
# -------------------------

# combined_probs is the weighted (file+chunk) probs, NOT renormalized
# compute absolute English mass (sum of combined_probs for english-like labels)
def is_label_english_like(lbl):
    lower = lbl.lower()
    if "english" in lower: return True
    if lower.startswith("en") or "en-" in lower or "en_" in lower or "en:" in lower: return True
    english_countries = ["america", "usa", "britain", "england", "uk", "india", "australia", "canada"]
    if any(c in lower for c in english_countries): return True
    return False

# english_total_abs = how much mass model assigns to English-like labels (0.0 - maybe << 1)
english_total_abs = sum(p for lbl, p in combined_probs.items() if is_label_english_like(lbl))

# Print debug absolute mass
print(f"\n🔎 Absolute English mass (model belief that audio is English-like): {english_total_abs*100:.4f}%")

# If english_total_abs is tiny, avoid forcing a bucket decision
MIN_ENGLISH_CONFIDENCE = 0.10  # require >=10% absolute english mass
if english_total_abs < MIN_ENGLISH_CONFIDENCE:
    print("\n⚠️ LOW CONFIDENCE. Please re-record reading these two sentences in one take:")
    print("  1) The quick brown fox jumps over the lazy dog near the riverbank.")
    print("  2) I bought new apples and oranges, but the unique choice surprised everyone.")
    # Optionally call your recording function again automatically
    # record_to_file("voice.wav", duration_s=15)
   # tuneable; 0.10 => require ≥10% absolute mass to trust English decision

if english_total_abs < MIN_ENGLISH_CONFIDENCE:
    print("\n⚠️ LOW CONFIDENCE: model assigns very little absolute probability to 'English'.")
    print("This means the model is uncertain — try these first:")
    print("  • Record 10–15s of natural spoken English (not stylized).")
    print("  • Ask the speaker to read two short sentences (recommended).")
    print("  • Ensure minimal background noise / no processing on the clip.")
    print("\nTop combined labels (debug):")
    for lbl, p in sorted(combined_probs.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {lbl} — {p*100:.6f}%")
    # don't attempt to map to an English accent — return safe message
    final_output = {
        "status": "low_confidence",
        "english_mass": english_total_abs,
        "top_labels": sorted(combined_probs.items(), key=lambda x: x[1], reverse=True)[:10]
    }
else:
    # Map to English buckets using combined_probs (absolute values), then compute relative shares
    english_buckets = {
        "American English": ["american", "united states", "usa", "us", "en-us"],
        "British English": ["british", "england", "uk", "united kingdom", "en-gb"],
        "Indian English": ["india", "indian", "in:", "ind"],
        "Australian English": ["australia", "australian", "au:"],
        "African English": ["africa", "nigeria", "ghana", "kenya", "south africa"],
        "Canadian English": ["canada", "canadian", "ca:"],
        "South American English": ["brazil", "argentina", "colombia", "chile"],
        "Other English": ["english", "en:"]
    }

    bucket_scores_abs = {k: 0.0 for k in english_buckets}
    for lbl, p in combined_probs.items():
        lower = lbl.lower()
        for bucket, keywords in english_buckets.items():
            if any(kw in lower for kw in keywords):
                bucket_scores_abs[bucket] += p
                break

    # If nothing mapped, use a fallback that assigns any 'en' labels to Other English
    if sum(bucket_scores_abs.values()) == 0:
        for lbl, p in combined_probs.items():
            if is_label_english_like(lbl):
                bucket_scores_abs["Other English"] += p

    # Compute relative percentages among English-like mass
    english_bucket_total_abs = sum(bucket_scores_abs.values())
    if english_bucket_total_abs <= 0:
        # fallback: map top combined label to Other English
        top_lbl, top_p = sorted(combined_probs.items(), key=lambda x: x[1], reverse=True)[0]
        bucket_scores_abs["Other English"] += top_p
        english_bucket_total_abs = sum(bucket_scores_abs.values())

    # Now compute results
    bucket_results = []
    for bucket, abs_score in bucket_scores_abs.items():
        if abs_score <= 0:
            continue
        rel_pct = (abs_score / english_bucket_total_abs) * 100.0    # relative among english-like labels
        abs_pct = abs_score * 100.0                                 # absolute model belief %
        bucket_results.append((bucket, abs_score, rel_pct, abs_pct))

    # Sort by absolute then relative if needed
    bucket_results.sort(key=lambda x: x[1], reverse=True)

    # Build final_output dict
    top_bucket, top_abs, top_rel, top_abs_pct = bucket_results[0]
    final_output = {
        "status": "ok",
        "english_mass_abs": english_total_abs,
        "detected_bucket": top_bucket,
        "detected_bucket_abs_pct": top_abs_pct,
        "detected_bucket_rel_pct_within_english": top_rel,
        "all_buckets": bucket_results,
        "top_combined_labels": sorted(combined_probs.items(), key=lambda x: x[1], reverse=True)[:12]
    }

    # Print friendly result
    print("\n🗂 English-accent estimation (safer):")
    print(f"  Detected: {top_bucket}")
    print(f"  Absolute model belief that this is English: {english_total_abs*100:.2f}%")
    print(f"  Confidence for {top_bucket}: {top_rel:.1f}% (relative among english-like labels)")
    print(f"  Absolute score for {top_bucket}: {top_abs_pct:.3f}% (model absolute probability mass)")

# final_output now has safe structured info you can show in UI or logs
