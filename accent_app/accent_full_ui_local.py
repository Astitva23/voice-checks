#!/usr/bin/env python3
"""
accent_full_local.py

- Loads local SpeechBrain accent model (no HF symlinks).
- Robust audio loading + resampling.
- Robust label parsing from label_encoder.txt / accent_encoder.txt (classic "'label' => idx" format).
- Attempts multiple classify methods and prints friendly named results.
- Edit MODEL_DIR and INPUT_FILE below for your paths.
"""

import os, sys, traceback, shutil
from pathlib import Path

# ========== USER CONFIG ==========
# Edit these two paths as needed for your machine
MODEL_DIR = r"E:\new-voice\voice-checks\accent_app\pretrained_models\accent-id-commonaccent_ecapa"
INPUT_FILE = r"E:\new-voice\voice-checks\accent_app\test_audio\voice.wav"
# ==================================

# Force HF to avoid symlinks and use a project-local cache
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
project_cache = os.path.join(os.path.dirname(__file__), ".hf_cache")
os.environ["HF_HOME"] = project_cache
os.environ["TRANSFORMERS_CACHE"] = project_cache
os.environ["HF_DATASETS_CACHE"] = project_cache

# -------------------------
# Basic path checks + ask-before-record flow
# -------------------------
if not os.path.isdir(MODEL_DIR):
    print(f"❌ Model folder not found: {MODEL_DIR}")
    sys.exit(1)

from pathlib import Path
INPUT_PATH = Path(INPUT_FILE)
RECORD_SR = 16000
RECORD_SECONDS = 6  # default, change if you want longer

def install_hint_and_exit():
    print("\n🔧 Missing dependency: 'sounddevice' is required to record from mic.")
    print("Install it in your venv with:")
    print("  pip install sounddevice")
    sys.exit(1)

def record_from_mic(dest_path, duration=RECORD_SECONDS, sr=RECORD_SR):
    try:
        import sounddevice as sd
        import soundfile as sf
        import numpy as np
    except Exception:
        install_hint_and_exit()

    dest_path = Path(dest_path)
    # delete previous temp voice files
    parent = dest_path.parent
    for f in parent.glob("voice*.wav"):
        try:
            f.unlink()
            print(f"ℹ️ Deleted old temp file: {f.name}")
        except Exception:
            pass

    print(f"\n🎤 Ready to record. Speak clearly into the microphone for ~{duration} seconds.")
    input("Press Enter to start recording...")
    try:
        print("⏺ Recording...")
        recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        audio = recording.squeeze()
        sf.write(str(dest_path), audio, sr, subtype='PCM_16')
        print(f"✅ Saved recording to: {dest_path} (sr={sr}, duration≈{duration}s)")
    except Exception as e:
        print("❌ Recording failed:", e)
        raise

# If file exists, ask the user what to do
if INPUT_PATH.exists():
    print(f"ℹ️ Found existing audio file: {INPUT_PATH}")
    print("Options:\n  [U]se existing file\n  [R]ecord a new sample (this will delete old temp files)\n  [Q]uit")
    choice = input("Choose (U/R/Q) [U]: ").strip().lower() or "u"
    if choice == "u":
        print("ℹ️ Using existing file.")
    elif choice == "r":
        try:
            record_from_mic(INPUT_PATH)
        except SystemExit:
            raise
        except Exception:
            print("❌ Failed to record. Exiting.")
            sys.exit(1)
    else:
        print("Exiting.")
        sys.exit(0)
else:
    # file missing -> ask to record
    print(f"ℹ️ No audio found at {INPUT_PATH}")
    yn = input("Would you like to record now? (Y/n) [Y]: ").strip().lower() or "y"
    if yn.startswith("y"):
        try:
            record_from_mic(INPUT_PATH)
        except SystemExit:
            raise
        except Exception:
            print("❌ Failed to record. Exiting.")
            sys.exit(1)
    else:
        print("No audio provided. Exiting.")
        sys.exit(0)



# Import heavy libs after env vars set
try:
    from speechbrain.inference import EncoderClassifier
except Exception:
    print("❌ Failed to import SpeechBrain. Ensure you're in the correct venv and 'speechbrain' is installed.")
    traceback.print_exc()
    sys.exit(1)

# -------------------------
# Label loader (robust) - UPDATED to ignore metadata lines like 'starting_index'
# -------------------------
import re, json
def load_labels_from_model_dir(model_dir, classifier_obj=None):
    model_dir = Path(model_dir)
    candidates = [
        model_dir / "label_encoder.txt",
        model_dir / "accent_encoder.txt",
        model_dir / "data" / "label_encoder.txt",
        model_dir / "data" / "accent_encoder.txt",
    ]
    label_map = {}
    for p in candidates:
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue

        parsed_any = False
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Skip obvious divider / metadata lines
            if line.startswith("=") or "starting_index" in line.lower() or line.lower().startswith("starting_index"):
                continue

            # "'label' => 0"
            m = re.match(r"""['"]([^'"]+)['"]\s*=>\s*([0-9]+)""", line)
            if m:
                lbl = m.group(1).strip()
                idx = int(m.group(2))
                # ignore metadata-like keys
                if lbl.lower().startswith("starting_index") or lbl.lower().startswith("===="):
                    continue
                label_map[idx] = lbl
                parsed_any = True
                continue

            # "label : index" or "label = index"
            m2 = re.match(r"""([^:=]+)[:=]\s*([0-9]+)""", line)
            if m2:
                lbl = m2.group(1).strip().strip("'\"")
                idx = int(m2.group(2))
                if lbl.lower().startswith("starting_index") or lbl.lower().startswith("===="):
                    continue
                label_map[idx] = lbl
                parsed_any = True
                continue

        if parsed_any:
            break

        # fallbacks
        try:
            obj = json.loads(text)
            if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
                return obj
        except Exception:
            pass

        if "," in text and len(text.split(",")) > 1:
            parts = [x.strip().strip("'\"") for x in text.split(",") if x.strip()]
            if parts:
                return parts

        lines = [ln.strip().strip("'\"") for ln in text.splitlines() if ln.strip()]
        if len(lines) > 1:
            return lines

    # convert label_map (index -> name) into ordered list
    if label_map:
        max_idx = max(label_map.keys())
        labels = [None] * (max_idx + 1)
        for idx, name in label_map.items():
            labels[idx] = name
        for i in range(len(labels)):
            if labels[i] is None:
                labels[i] = f"label_{i}"
        return labels

    # Try to extract from classifier object if provided
    if classifier_obj is not None:
        try:
            h = getattr(classifier_obj, "hparams", None)
            if h and "label_encoder" in h:
                val = h["label_encoder"]
                try:
                    return list(val)
                except Exception:
                    pass
        except Exception:
            pass

    return None

# -------------------------
# Robust audio classification helper
# -------------------------
def classify_audio_robust(classifier, filepath, target_sr=16000):
    import soundfile as sf
    import numpy as np
    import torch
    import librosa

    # 1) direct convenience wrapper
    try:
        print("Attempt 1: classifier.classify_file(...)")
        out = classifier.classify_file(filepath)
        print("-> classify_file succeeded")
        return out
    except Exception as e1:
        print("Attempt 1 failed:", repr(e1))

    # 2) load with soundfile (preferred), fallback to torchaudio
    data = None; sr = None
    try:
        print("Attempt 2: loading with soundfile...")
        data, sr = sf.read(filepath, always_2d=False)
        if hasattr(data, "ndim") and data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        print(f"Loaded: shape={data.shape}, sr={sr}")
    except Exception as e2:
        print("soundfile load failed:", repr(e2))
        try:
            import torchaudio
            print("Attempt 2b: loading with torchaudio...")
            waveform, sr = torchaudio.load(filepath)
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=0)
            data = waveform.cpu().numpy().astype(np.float32)
            print(f"torchaudio loaded: shape={data.shape}, sr={sr}")
        except Exception as e2b:
            print("torchaudio load also failed:", repr(e2b))
            raise RuntimeError("Failed to load audio with soundfile and torchaudio") from e2b

    # 3) resample if needed
    if sr != target_sr:
        try:
            print(f"Resampling {sr} -> {target_sr} using librosa...")
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
            print("Resample done. New length:", data.shape)
        except Exception as e_rs:
            print("librosa resample failed:", repr(e_rs))
            try:
                import torch, torchaudio
                tensor = torch.from_numpy(data).unsqueeze(0)
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                tensor_rs = resampler(tensor)
                data = tensor_rs.squeeze(0).cpu().numpy()
                sr = target_sr
                print("torchaudio resample succeeded.")
            except Exception as e_rs2:
                print("torchaudio resample failed too:", repr(e_rs2))

    # 4) try classify_batch and other call styles
    last_exc = None
    try_inputs = []
    import torch
    try_inputs.append(("numpy_1d", data))
    try_inputs.append(("numpy_list", [data]))
    try_inputs.append(("torch_1d", torch.from_numpy(data)))
    try_inputs.append(("torch_batched", torch.from_numpy(data).unsqueeze(0)))
    try_inputs.append(("pylist", data.tolist()))

    for name, inp in try_inputs:
        try:
            print(f"Attempt classify with input type: {name}")
            if hasattr(classifier, "classify_batch"):
                try:
                    out = classifier.classify_batch(inp)
                    print(f"-> classify_batch accepted input: {name}")
                    return out
                except Exception as e_cb:
                    print(f" classify_batch failed for {name}: {repr(e_cb)}")
                    last_exc = e_cb
            try: 
                out = classifier.classify_file(inp)  # some wrappers accept arrays
                print(f"-> classify_file accepted non-path input: {name}")
                return out
            except Exception as e_cf:
                print(f" classify_file(non-path) failed for {name}: {repr(e_cf)}")
                last_exc = e_cf
        except Exception as e_any:
            print(f"Unexpected failure trying input {name}: {repr(e_any)}")
            last_exc = e_any

    print("All classify attempts failed. Raising last exception.")
    raise last_exc or RuntimeError("Unknown classification failure")

# -------------------------
# Load model (local)
# -------------------------
print("Loading model from local folder (no symlinks):", MODEL_DIR)
try:
    classifier = EncoderClassifier.from_hparams(source=MODEL_DIR, savedir=MODEL_DIR)
except Exception:
    print("❌ Model load failed. Traceback:")
    traceback.print_exc()
    sys.exit(1)

# -------------------------
# Load labels
# -------------------------
labels = load_labels_from_model_dir(MODEL_DIR, classifier_obj=classifier)
if labels:
    print("✅ Labels loaded (index -> name):")
    for i, nm in enumerate(labels):
        print(f"  {i}: {nm}")
else:
    print("⚠️ Could not auto-load full label list. Predictions will show indices. Consider adding a proper label_encoder.txt")

# -------------------------
# Classify file (robust)
# -------------------------
print("\nClassifying:", INPUT_FILE)
try:
    out = classify_audio_robust(classifier, INPUT_FILE)
except Exception:
    print("❌ classify_file failed. Traceback:")
    traceback.print_exc()
    sys.exit(1)

# -------------------------
# Parse & display results
# -------------------------
import torch, numpy as np
print("\n🔧 Debug: raw classifier output (for troubleshooting):")
print(out)

# normalize many possible output formats
logits = None
if isinstance(out, (tuple, list)) and len(out) >= 1:
    first = out[0]
    if hasattr(first, "squeeze"):
        logits = first.squeeze()
    else:
        try:
            logits = torch.tensor(first)
        except Exception:
            logits = None
else:
    if hasattr(out, "squeeze"):
        logits = out.squeeze()

if logits is None:
    print("❌ Could not interpret model output (no logits). Raw output printed above.")
    sys.exit(1)

if not isinstance(logits, torch.Tensor):
    logits = torch.tensor(logits)

probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

if labels and len(labels) == len(probs):
    topk_idx = np.argsort(probs)[-5:][::-1]
    print("\n🔎 Top predictions:")
    for i in topk_idx:
        print(f"  {labels[i]} — {probs[i]*100:.2f}%")
    best = int(np.argmax(probs))
    print(f"\n🎯 Final Guess: {labels[best]} — {probs[best]*100:.2f}%")
else:
    # fallback raw indices
    topk_idx = np.argsort(probs)[-10:][::-1]
    print("\n🔎 Top raw predictions (index -> prob):")
    for i in topk_idx:
        print(f"  idx {i} -> {probs[i]*100:.6f}%")
    if labels:
        print("\n⚠️ Label count does not match probs length. Label sample (first 20):", labels[:20])
    else:
        print("\n⚠️ No labels available to map indices to names.")