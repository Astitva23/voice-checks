# english_accent_quick.py
import os, sys
import numpy as np
import soundfile as sf
import torchaudio
import torch
from speechbrain.pretrained import EncoderClassifier
# ====== PATCH: change config values ======
chunk_seconds = 12           # longer chunks
hop_seconds = 3              # more overlap -> smoother aggregation
target_sr = 16000
min_speech_seconds = 6.0     # require at least 6s of speech after trimming

# -------------------------
# Helpers: trim silence, normalize
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

# -------------------------
# Load model (SpeechBrain VoxLingua107)
# -------------------------
print("🧠 Loading VoxLingua107 model (SpeechBrain) - may download on first run...")
try:
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-voxlingua107-ecapa",
        savedir="pretrained_models/lang-id-voxlingua107-ecapa"
    )
except Exception as e:
    print("❌ Failed to load model:", e)
    print("See instructions in the script header about HF symlink or running terminal as admin.")
    sys.exit(1)
print("✅ Model loaded.\n")

# -------------------------
# Input check + preprocess
# -------------------------
if not os.path.exists(wav_in):
    print(f"❗ File '{wav_in}' not found. Record and save as '{wav_in}' next to this script.")
    sys.exit(1)

y, sr = sf.read(wav_in)
if y.ndim > 1:
    y = y.mean(axis=1)

y = trim_silence(y, sr, top_db=30)
y = normalize_rms(y, target_db=-20.0)

if len(y)/sr < min_speech_seconds:
    print(f"🔊 Audio too short after trimming ({len(y)/sr:.2f}s). Please record 8-12s of speech.")
    sys.exit(0)

sf.write(clean_path, y, sr)

# load into torchaudio tensor
signal, fs = torchaudio.load(clean_path)  # (channels, samples)
if signal.shape[0] > 1:
    signal = torch.mean(signal, dim=0, keepdim=True)

# resample to model sample rate
if fs != target_sr:
    resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=target_sr)
    signal = resampler(signal)
    fs = target_sr

signal = signal.float()

# -------------------------
# Chunking strategy
# -------------------------
samples_per_chunk = int(chunk_seconds * fs)
hop = int(hop_seconds * fs)
total_samples = signal.shape[1]
chunks = []
start = 0
while start < total_samples:
    end = min(total_samples, start + samples_per_chunk)
    chunk = signal[:, start:end]
    # only keep chunk with at least 1s
    if chunk.shape[1] >= int(1.0 * fs):
        # if chunk shorter than chunk size, pad with zeros (model accepts variable length but padding keeps consistency)
        if chunk.shape[1] < samples_per_chunk:
            pad = torch.zeros((1, samples_per_chunk - chunk.shape[1]))
            chunk = torch.cat([chunk, pad], dim=1)
        chunks.append(chunk)
    start += hop

if len(chunks) == 0:
    print("❗ No valid audio chunks extracted.")
    sys.exit(1)

# tiny music detector: returns True if chunk likely musical (high spectral centroid + high spectral flux)
import librosa

def is_musical(y_np, sr):
    # y_np: 1D numpy float audio
    if len(y_np) < sr:  # <1s -> don't classify as music
        return False
    z = np.abs(librosa.stft(y_np.astype(np.float32), n_fft=1024, hop_length=512))
    spec_cent = np.mean(librosa.feature.spectral_centroid(S=z, sr=sr))
    spec_flux = np.mean(np.sum(np.diff(z, axis=1)**2, axis=0))
    # threshold heuristics — tune if needed
    if spec_cent > 3000 and spec_flux > 1e6:
        return True
    return False




# --------- Robust classify + aggregate by label name ----------
from collections import defaultdict

print(f"🔊 Classifying {len(chunks)} chunk(s)... (chunk size {chunk_seconds}s, hop {hop_seconds}s)")
label_prob_sums = defaultdict(float)   # sum of probs for each label across successful chunks
successful_chunks = 0
failed_chunks = 0
seen_label_sets = []

for idx, chunk in enumerate(chunks):
    try:
        out = classifier.classify_batch(chunk)
    except Exception as e:
        # fallback to classify_file only once if batch fails for this chunk
        try:
            out = classifier.classify_file(clean_path)
        except Exception as e2:
            print(f"⚠️ Chunk {idx}: classification failed (batch & file). Skipping chunk.\n  Errors: {e} ; {e2}")
            failed_chunks += 1
            continue

    # expect (logits, ..., ..., labels)
    if isinstance(out, (list, tuple)) and len(out) >= 4:
        logits = out[0].squeeze()
        labels = list(out[3])
    else:
        print(f"⚠️ Chunk {idx}: unexpected model output format {type(out)}. Skipping chunk.")
        failed_chunks += 1
        continue

    # convert to probs safely
    try:
        probs = torch.softmax(logits, dim=0).cpu().numpy()
    except Exception as e:
        print(f"⚠️ Chunk {idx}: failed to convert logits -> probs: {e}. Skipping chunk.")
        failed_chunks += 1
        continue

    # Sanity: if labels length differs from probs length, align by min (but use names)
    L = min(len(labels), probs.shape[0])
    labels = labels[:L]
    probs = probs[:L]

    # accumulate by label name
    for lbl, p in zip(labels, probs):
        label_prob_sums[lbl] += float(p)

    successful_chunks += 1
    seen_label_sets.append(labels)

if successful_chunks == 0:
    print(f"❌ All chunks failed ({failed_chunks}/{len(chunks)}). Cannot classify.")
    sys.exit(1)

# compute mean prob per label across chunks (treat missing as 0)
mean_probs = {lbl: label_prob_sums[lbl] / successful_chunks for lbl in label_prob_sums}

# to display top-k, normalize or sort by mean_probs
sorted_labels = sorted(mean_probs.items(), key=lambda x: x[1], reverse=True)
topk = sorted_labels[:10]

print("\n🔎 Raw model top predictions (aggregated over chunks):")
for lbl, p in topk[:5]:
    print(f"  {lbl} — {p*100:.2f}%")

# create arrays for downstream code (if you expect arrays)
label_list = [lbl for lbl, _ in sorted_labels]
accum_probs = np.array([p for _, p in sorted_labels])

# continue with your English-bucket mapping using label_list and accum_probs
# ------------------------------------------------------------------------


# -------------------------
# Print raw top-5 labels (helpful for debugging)
# -------------------------
topk_idx = np.argsort(accum_probs)[-5:][::-1]
print("\n🔎 Raw model top-5 language/variety predictions (from VoxLingua):")
for i in topk_idx:
    lbl = label_list[i] if i < len(label_list) else f"label_{i}"
    p = accum_probs[i]
    print(f"  {lbl} — {p*100:.2f}%")


# -------------------------
# Heuristic mapping to English-accent buckets
# -------------------------
# Map known keywords in VoxLingua label strings to English accent buckets.
# This mapping is heuristic — you can edit keywords for your needs.
# ====== More exhaustive bucket mapping (replace previous mapping) ======
english_buckets = {
    "American English": ["american", "united states", "usa", "us", "united_states", "en-us", "us:","unitedstates"],
    "British English": ["british", "england", "uk", "united kingdom", "england:","en-gb","uk:"],
    "Indian English": ["india", "indian", "in:", "ind", "india:"],
    "Australian English": ["australia", "australian", "au:","australia:"],
    "African English": ["africa", "nigeria", "ghana", "kenya", "south africa", "south_africa", "ng:", "za:"],
    "Canadian English": ["canada", "canadian", "ca:"],
    "South American English": ["brazil", "argentina", "colombia", "chile", "south america", "br:"],
    "Other English": ["caribbean", "philippines", "singapore", "singapore:"]
}

# Create a lowercase mapping loop
bucket_scores = {k: 0.0 for k in english_buckets.keys()}
english_total = 0.0

for i, lbl in enumerate(label_list):
    lower = lbl.lower()
    prob = float(accum_probs[i])
    for bucket, keywords in english_buckets.items():
        for kw in keywords:
            if kw in lower:
                bucket_scores[bucket] += prob
                english_total += prob
                break

# -------------------------
# If the model didn't directly map many labels to English, we can fallback:
# heuristic: if label contains 'english' or 'en:' anywhere => map to Other English bucket
if english_total < 0.1:
    for i, lbl in enumerate(label_list):
        lower = lbl.lower()
        prob = float(accum_probs[i])
        if "english" in lower or lower.startswith("en") or "en:" in lower:
            bucket_scores["Other English"] += prob
            english_total += prob

# -------------------------
# Present English-focused results
# -------------------------
print("\n🗂 English-accent estimation (heuristic aggregation):")
if english_total <= 0.0:
    print("  ⚠️ Low English signal: the model thinks the speaker's L1 may be non-English or confidence is low.")
else:
    # Normalize bucket scores to sum to 1 within the english_total
    sorted_buckets = sorted(bucket_scores.items(), key=lambda x: x[1], reverse=True)
    for bucket, score in sorted_buckets:
        if score <= 0:
            continue
        conf = score / english_total * 100.0
        print(f"  {bucket} — {conf:.1f}% (relative among detected English labels)")

    # show top English bucket as the detected accent
    top_bucket, top_score = sorted_buckets[0]
    if top_score > 0:
        conf_pct = top_score / english_total * 100.0
        print(f"\n✅ Final guess: {top_bucket} (confidence {conf_pct:.1f}% among English-like labels)")
    else:
        print("\n⚠️ No strong English-like label found.")

print("\nDone.")
