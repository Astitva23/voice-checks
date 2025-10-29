# run_accent_local_nosymlink.py
# Safe loader: disables HF symlinks inside this process and forces local-only load.
# Put this file in your project root and run: python run_accent_local_nosymlink.py

import os
import sys

# ======== Force no HF symlinks and use local cache inside project ========
# Must set BEFORE importing anything that may touch huggingface_hub or speechbrain
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Use a project-local HF cache (avoids permissions in user profile)
local_hf_cache = os.path.join(os.getcwd(), ".hf_cache")
os.environ["HF_HOME"] = local_hf_cache      # huggingface hub uses this
os.environ["TRANSFORMERS_CACHE"] = local_hf_cache
os.environ["HF_DATASETS_CACHE"] = local_hf_cache

# optional: make python show full tracebacks
import traceback

# ======== Config - edit if your paths differ ========
LOCAL_MODEL_DIR = r"E:\new-voice\voice-checks\accent_app\pretrained_models\accent-id-commonaccent_ecapa"
INPUT_FILE = r"E:\new-voice\voice-checks\accent_app\test_audio\voice.wav"

# quick checks before heavy imports
if not os.path.exists(LOCAL_MODEL_DIR):
    print("❌ Local model folder missing:", LOCAL_MODEL_DIR)
    print("Place the model files (hyperparams.yaml, classifier.ckpt, embedding_model.ckpt, accent_encoder.txt) there.")
    sys.exit(1)

if not os.path.exists(INPUT_FILE):
    print("❌ Input audio missing:", INPUT_FILE)
    sys.exit(1)

# ensure label file exists in some common names - copy if alternate present
possible = [
    os.path.join(LOCAL_MODEL_DIR, "label_encoder.txt"),
    os.path.join(LOCAL_MODEL_DIR, "accent_encoder.txt"),
    os.path.join(LOCAL_MODEL_DIR, "accent_encoder"),
]
for p in possible:
    if os.path.exists(p) and not os.path.exists(os.path.join(LOCAL_MODEL_DIR, "label_encoder.txt")):
        try:
            import shutil
            shutil.copy2(p, os.path.join(LOCAL_MODEL_DIR, "label_encoder.txt"))
            print("ℹ️ Copied", p, "-> label_encoder.txt")
            break
        except Exception as e:
            print("Warning: could not copy label file:", e)

# ======== Now import heavy libs (after env vars set) ========
try:
    from speechbrain.inference import EncoderClassifier
except Exception as e:
    print("❌ Failed to import SpeechBrain. Full traceback:")
    traceback.print_exc()
    sys.exit(1)

# ======== Load classifier from LOCAL_MODEL_DIR (no HF cache symlink) ========
print("🧠 Loading model from local folder (no symlinks) ...")
try:
    classifier = EncoderClassifier.from_hparams(source=LOCAL_MODEL_DIR, savedir=LOCAL_MODEL_DIR)
except Exception as e:
    print("❌ Model load failed (from_hparams). Full traceback:")
    traceback.print_exc()
    print("\nHint: if this still errors with WinError 1314, try running this script once as Administrator.")
    sys.exit(1)

# ======== Classify the file ========
# ===== Robust classification helper =====
import soundfile as sf
import librosa

def classify_audio_robust(classifier, filepath, target_sr=16000):
    """
    Try multiple ways to classify the audio:
      1) classifier.classify_file(filepath)
      2) load via soundfile (or torchaudio fallback) -> try classifier.classify_batch or classifier.classify_batch([tensor])
      3) try other input shapes (numpy array, torch.tensor, list)
    Returns the raw classifier output (whatever works) or raises the last exception.
    """
    import os, traceback, torch, numpy as np

    # 1) direct convenience wrapper
    try:
        print("Attempt 1: classifier.classify_file(...)")
        out = classifier.classify_file(filepath)
        print("-> classify_file succeeded")
        return out
    except Exception as e1:
        print("Attempt 1 failed:", repr(e1))

    # 2) load audio using soundfile (preferred) and resample if needed
    try:
        print("Attempt 2: loading with soundfile...")
        data, sr = sf.read(filepath, always_2d=False)
        # soundfile returns shape (n,) or (n, channels) depending; make mono
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        data = data.astype(np.float32)
        print(f"Loaded: shape={data.shape}, sr={sr}")
    except Exception as e2:
        print("soundfile load failed:", repr(e2))
        # fallback: try torchaudio
        try:
            import torchaudio
            print("Attempt 2b: loading with torchaudio...")
            waveform, sr = torchaudio.load(filepath)
            # waveform shape: (channels, samples)
            if waveform.ndim > 1:
                waveform = torch.mean(waveform, dim=0)  # (samples,)
            data = waveform.cpu().numpy().astype(np.float32)
            print(f"torchaudio loaded: shape={data.shape}, sr={sr}")
        except Exception as e2b:
            print("torchaudio load also failed:", repr(e2b))
            raise RuntimeError("Failed to load audio with soundfile and torchaudio") from e2b

    # 3) resample if needed (librosa)
    if sr != target_sr:
        try:
            print(f"Resampling {sr} -> {target_sr} using librosa...")
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
            print("Resample done. New length:", data.shape)
        except Exception as e_rs:
            print("librosa resample failed:", repr(e_rs))
            # try torchaudio resample as fallback
            try:
                import torch, torchaudio
                tensor = torch.from_numpy(data).unsqueeze(0)  # (1, samples)
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                tensor_rs = resampler(tensor)
                data = tensor_rs.squeeze(0).cpu().numpy()
                sr = target_sr
                print("torchaudio resample succeeded.")
            except Exception as e_rs2:
                print("torchaudio resample failed too:", repr(e_rs2))
                # continue with original sr (some models can accept different sr) 

    # 4) prepare candidate inputs and try classify_batch / classify_batch-like calls
    last_exc = None
    try_inputs = []

    # a) numpy 1D
    try_inputs.append(("numpy_1d", data))
    # b) numpy list
    try_inputs.append(("numpy_list", [data]))
    # c) torch 1D tensor
    try_inputs.append(("torch_1d", torch.from_numpy(data)))
    # d) torch batched (1, samples)
    try_inputs.append(("torch_batched", torch.from_numpy(data).unsqueeze(0)))
    # e) float32 python list
    try_inputs.append(("pylist", data.tolist()))

    for name, inp in try_inputs:
        try:
            print(f"Attempt classify with input type: {name}")
            # Many SpeechBrain wrappers accept numpy array or torch tensor
            # try classify_batch first
            if hasattr(classifier, "classify_batch"):
                try:
                    out = classifier.classify_batch(inp)
                    print(f"-> classify_batch accepted input: {name}")
                    return out
                except Exception as e_cb:
                    print(f" classify_batch failed for {name}: {repr(e_cb)}")
                    last_exc = e_cb
            # fallback: some wrappers accept a list of signals or a waveform array
            try:
                out = classifier.classify_file(inp)  # some implementations tolerate an array (rare)
                print(f"-> classify_file accepted non-path input: {name}")
                return out
            except Exception as e_cf:
                print(f" classify_file(non-path) failed for {name}: {repr(e_cf)}")
                last_exc = e_cf

        except Exception as e_any:
            print(f"Unexpected failure trying input {name}: {repr(e_any)}")
            last_exc = e_any

    # if nothing succeeded, raise last exception to show the real traceback
    print("All classify attempts failed. Raising last exception.")
    raise last_exc or RuntimeError("Unknown classification failure")

    

# ===== Replace the old classify try/except with the robust helper =====
print("✅ Model loaded. Classifying (robust)...", INPUT_FILE)
try:
    out = classify_audio_robust(classifier, INPUT_FILE)
except Exception as e:
    print("❌ classify_file failed. Traceback:")
    import traceback
    traceback.print_exc()
    sys.exit(1)

    

# ======== Robust result parsing & display (replace previous parse block) ========
import torch
import numpy as np

print("\n🔧 Debug: raw classifier output (for troubleshooting):")
print(out)   # prints the same raw tuple you previously saw

# Normalise different possible output formats from SpeechBrain
logits = None
labels = None

# Common SpeechBrain tuple: (logits_tensor, score_tensor, idx_tensor, labels_list)
if isinstance(out, (tuple, list)) and len(out) >= 1:
    # if first element is a tensor of logits
    first = out[0]
    if hasattr(first, "squeeze"):
        logits = first.squeeze()
    # try to obtain labels list (often at index 3)
    if len(out) >= 4 and isinstance(out[3], (list, tuple)):
        labels = list(out[3])

# If still no logits, maybe out itself is a tensor
if logits is None:
    if hasattr(out, "squeeze"):
        logits = out.squeeze()

# If still no labels, try to read from model (best-effort)
if labels is None:
    try:
        # some EncoderClassifier objects expose id2label or label_encoder
        cfg = getattr(classifier, "hparams", None)
        if cfg and "label_encoder" in cfg:
            labels = cfg["label_encoder"]
    except Exception:
        labels = None

# Final safety check
if logits is None:
    print("❌ Could not interpret model output (no logits). Raw output above.")
    sys.exit(1)

# Ensure logits is a torch tensor
if not isinstance(logits, torch.Tensor):
    logits = torch.tensor(logits)

# Softmax to probabilities
probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

# If we have labels and lengths match, show names
if labels and len(labels) == len(probs):
    topk_idx = np.argsort(probs)[-5:][::-1]
    print("\n🔎 Top predictions:")
    for i in topk_idx:
        print(f"  {labels[i]} — {probs[i]*100:.2f}%")
    best = int(np.argmax(probs))
    print(f"\n🎯 Final Guess: {labels[best]} — {probs[best]*100:.2f}%")
else:
    # fallback: show top indices and their probs and any available label mapping
    topk_idx = np.argsort(probs)[-10:][::-1]
    print("\n🔎 Top raw predictions (index -> prob):")
    for i in topk_idx:
        print(f"  idx {i} -> {probs[i]*100:.6f}%")
    if labels:
        print("\n⚠️ Label count does not match probs length.")
        print("Label sample (first 20):", labels[:20])
    else:
        print("\n⚠️ No labels available to map indices to names. If you want names, ensure your model folder has `accent_encoder.txt` or `label_encoder.txt`.")
