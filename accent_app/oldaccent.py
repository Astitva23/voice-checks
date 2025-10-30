#!/usr/bin/env python3
"""
accent_full_local.py

Single-file helper for:
 - robustly loading local SpeechBrain accent classifier (no HF symlinks)
 - robust label parsing + permanent cleaning (backs up original file, writes cleaned)
 - robust audio loading/resampling
 - classify helper used by app_flask.py and CLI ask-to-record flow

Edit MODEL_DIR and INPUT_FILE below if needed.
"""

import os, sys, traceback, json, re, shutil
from pathlib import Path

# ========== USER CONFIG ========== (edit as needed)
MODEL_DIR = r"E:\new-voice\voice-checks\accent_app\pretrained_models\accent-id-commonaccent_ecapa"
INPUT_FILE = r"E:\new-voice\voice-checks\accent_app\test_audio\voice.wav"
# ==================================

# Avoid HF symlink issues
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
project_cache = os.path.join(os.path.dirname(__file__), ".hf_cache")
os.environ["HF_HOME"] = project_cache
os.environ["TRANSFORMERS_CACHE"] = project_cache
os.environ["HF_DATASETS_CACHE"] = project_cache

# -------------------------
# Robust label loader + permanent cleaner
# -------------------------
def parse_label_file_text(text):
    """Return list of parsed labels (ordered by index) or appearance list."""
    label_map = {}
    appearance = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        low = line.lower()
        if line.startswith("=") or low.startswith("starting_index") or low.startswith("==="):
            continue
        # 'label' => 0
        m = re.match(r"""['"]([^'"]+)['"]\s*=>\s*([0-9]+)""", line)
        if m:
            lbl = m.group(1).strip()
            idx = int(m.group(2))
            if lbl.lower().startswith("starting_index"):
                continue
            label_map[idx] = lbl
            if lbl not in appearance:
                appearance.append(lbl)
            continue
        # label : index  or label = index
        m2 = re.match(r"""['"]?([^'":=]+)['"]?\s*[:=]\s*([0-9]+)""", line)
        if m2:
            lbl = m2.group(1).strip().strip("'\"")
            idx = int(m2.group(2))
            if lbl.lower().startswith("starting_index"):
                continue
            label_map[idx] = lbl
            if lbl not in appearance:
                appearance.append(lbl)
            continue
        # fallback: plain label per-line
        if "=>" not in line and ":" not in line and "=" not in line and len(line) < 80:
            candidate = line.strip().strip("'\"")
            if candidate and not candidate.lower().startswith("starting_index") and candidate not in appearance:
                appearance.append(candidate)
    # build labels list if label_map present
    if label_map:
        max_idx = max(label_map.keys())
        labels = [None] * (max_idx + 1)
        for idx, nm in label_map.items():
            labels[idx] = nm
        # fill holes with appearance or placeholders
        ai = 0
        for i in range(len(labels)):
            if labels[i] is None:
                while ai < len(appearance) and appearance[ai] in labels:
                    ai += 1
                if ai < len(appearance):
                    labels[i] = appearance[ai]; ai += 1
                else:
                    labels[i] = f"label_{i}"
        return labels
    if appearance:
        return appearance
    return None

def load_labels_from_model_dir(model_dir, classifier_obj=None, write_cleaned=True, overwrite_original=True):
    """
    Robustly parse label encoder files. If found, writes label_encoder.cleaned.txt
    and (by default) overwrites label_encoder.txt with cleaned contents after backup.
    Returns labels list (index -> name) or None.
    """
    model_dir = Path(model_dir)
    candidates = [
        model_dir / "label_encoder.txt",
        model_dir / "accent_encoder.txt",
        model_dir / "data" / "label_encoder.txt",
        model_dir / "data" / "accent_encoder.txt",
    ]
    found_text = None
    src_path = None
    for p in candidates:
        if p.exists():
            try:
                t = p.read_text(encoding="utf-8", errors="replace")
                if t.strip():
                    found_text = t
                    src_path = p
                    break
            except Exception:
                continue
    # fallback: try reading label_encoder.cleaned.txt if present
    cleaned_path = model_dir / "label_encoder.cleaned.txt"
    if found_text is None and cleaned_path.exists():
        try:
            t = cleaned_path.read_text(encoding="utf-8", errors="replace")
            if t.strip():
                found_text = t
                src_path = cleaned_path
        except Exception:
            pass

    # Try to parse
    if found_text:
        labels = parse_label_file_text(found_text)
        if labels:
            # Write cleaned file
            if write_cleaned:
                try:
                    cleaned_text = "\n".join([str(x) for x in labels])
                    cleaned_path.write_text(cleaned_text, encoding="utf-8")
                except Exception as e:
                    print("⚠️ Could not write cleaned label file:", e)
            # Overwrite original label_encoder.txt safely (backup)
            if overwrite_original and src_path is not None and src_path.exists():
                try:
                    orig = model_dir / (src_path.name + ".orig_backup")
                    if not orig.exists():
                        shutil.copy2(src_path, orig)
                        print(f"🔁 Backed up original label file to: {orig}")
                    # overwrite the canonical label_encoder.txt (create if missing)
                    target = model_dir / "label_encoder.txt"
                    cleaned_text = "\n".join([str(x) for x in labels])
                    target.write_text(cleaned_text, encoding="utf-8")
                    print(f"✅ Wrote cleaned labels to: {target}")
                except Exception as e:
                    print("⚠️ Could not overwrite original label file:", e)
            return labels
    # fallback: try classifier hparams
    if classifier_obj is not None:
        try:
            h = getattr(classifier_obj, "hparams", None)
            if h:
                for key in ("label_encoder", "labels", "id2label", "label2id"):
                    if key in h:
                        val = h[key]
                        try:
                            if isinstance(val, dict):
                                inv = {int(v): k for k, v in val.items()}
                                max_idx = max(inv.keys())
                                labels = [inv.get(i, f"label_{i}") for i in range(max_idx+1)]
                                return labels
                            if isinstance(val, (list, tuple)):
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
    """
    Try multiple loading/classify strategies:
     - classifier.classify_file(filepath)
     - read with soundfile (sf), resample with librosa if needed
     - torchaudio load fallback
     - attempt classify_batch with various input shapes
    Returns model output (logits/tuple) on success or raises.
    """
    import soundfile as sf
    import numpy as np
    import torch

    # 1) try direct convenience wrapper (fastest)
    try:
        out = classifier.classify_file(filepath)
        return out
    except Exception as e1:
        # proceed to robust loading
        pass

    # 2) load with soundfile
    data = None; sr = None
    try:
        data, sr = sf.read(filepath, always_2d=False)
        if hasattr(data, "ndim") and data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
    except Exception as e_sf:
        # fallback torchaudio
        try:
            import torchaudio
            waveform, sr = torchaudio.load(filepath)
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=0)
            data = waveform.cpu().numpy().astype(np.float32)
        except Exception as e_ta:
            raise RuntimeError("Failed to load audio with soundfile and torchaudio") from e_ta

    # 3) resample if needed
    if sr != target_sr:
        try:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except Exception:
            try:
                import torchaudio, torch
                tensor = torch.from_numpy(data).unsqueeze(0)
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                tensor_rs = resampler(tensor)
                data = tensor_rs.squeeze(0).cpu().numpy()
                sr = target_sr
            except Exception:
                pass

    # 4) Try classify_batch and classify_file with different input shapes
    last_exc = None
    import torch
    try_inputs = [
        ("numpy_1d", data),
        ("numpy_list", [data]),
        ("torch_1d", torch.from_numpy(data)),
        ("torch_batched", torch.from_numpy(data).unsqueeze(0)),
        ("pylist", data.tolist()),
    ]
    for name, inp in try_inputs:
        try:
            if hasattr(classifier, "classify_batch"):
                try:
                    out = classifier.classify_batch(inp)
                    return out
                except Exception:
                    pass
            try:
                out = classifier.classify_file(inp)
                return out
            except Exception:
                pass
        except Exception as e_any:
            last_exc = e_any
    if last_exc:
        raise last_exc
    raise RuntimeError("Unknown classify failure")

# -------------------------
# Load model (local) - exported object
# -------------------------
print("Loading model (local) from:", MODEL_DIR)
classifier = None
try:
    # import lazy to avoid import-time errors in other contexts
    from speechbrain.inference import EncoderClassifier as SBEncoder
    classifier = SBEncoder.from_hparams(source=MODEL_DIR, savedir=MODEL_DIR)
    print("✅ Model loaded.")
except Exception:
    print("❌ Failed to import/load the SpeechBrain model. Traceback:")
    traceback.print_exc()
    # leave classifier as None but allow functions to attempt lazy load later

# Ensure labels variable always exists and perform permanent clean
try:
    labels = load_labels_from_model_dir(MODEL_DIR, classifier_obj=classifier, write_cleaned=True, overwrite_original=True)
except Exception as e:
    print("⚠️ Label loading/cleaning failed:", e)
    labels = None

if not labels:
    # fallback: try to read cleaned file or original label file
    try:
        p = Path(MODEL_DIR) / "label_encoder.cleaned.txt"
        if not p.exists():
            p = Path(MODEL_DIR) / "label_encoder.txt"
        if p.exists():
            labels = [ln.strip() for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip() and not ln.lower().startswith("starting_index")]
    except Exception:
        labels = []
print(f"Loaded labels: {len(labels)} entries.")

# -------------------------
# CLI ask-before-record flow (when run directly)
# -------------------------
if __name__ == "__main__":
    import sounddevice as sd, soundfile as sf, numpy as np
    RECORD_SR = 16000
    RECORD_SECONDS = 6
    INPUT_PATH = Path(INPUT_FILE)
    INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    def record_to_file(dest_path, duration=RECORD_SECONDS, sr=RECORD_SR):
        dest_path = Path(dest_path)
        # delete previous voice*.wav in folder
        for f in dest_path.parent.glob("voice*.wav"):
            try: f.unlink()
            except: pass
        print(f"\n🎤 Ready to record. Speak clearly for ~{duration} seconds.")
        input("Press Enter to start recording...")
        arr = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
        sd.wait()
        audio = arr.squeeze()
        sf.write(str(dest_path), audio, sr, subtype="PCM_16")
        print("✅ Saved:", dest_path)

    # basic path checks
    if not Path(MODEL_DIR).is_dir():
        print("❌ Model folder not found:", MODEL_DIR)
        sys.exit(1)

    if INPUT_PATH.exists():
        print("ℹ️ Found existing audio:", INPUT_PATH)
        choice = input("Options: [U]se existing, [R]ecord new, [Q]uit (default U): ").strip().lower() or "u"
        if choice == "u":
            print("Using existing file.")
        elif choice == "r":
            try:
                record_to_file(INPUT_PATH)
            except Exception as e:
                print("Recording failed:", e); sys.exit(1)
        else:
            print("Quitting."); sys.exit(0)
    else:
        yn = input(f"No audio found at {INPUT_PATH}. Record now? (Y/n) [Y]: ").strip().lower() or "y"
        if yn.startswith("y"):
            try:
                record_to_file(INPUT_PATH)
            except Exception as e:
                print("Recording failed:", e); sys.exit(1)
        else:
            print("No audio provided. Exiting."); sys.exit(0)

    # classify and print result
    try:
        # lazy load if classifier none
        if classifier is None:
            from speechbrain.inference import EncoderClassifier as SBEnc
            classifier = SBEnc.from_hparams(source=MODEL_DIR, savedir=MODEL_DIR)
        out = classify_audio_robust(classifier, str(INPUT_PATH))
        # parse logits and print probabilities
        import torch, numpy as np
        if isinstance(out, (list, tuple)) and len(out) >= 1:
            logits = out[0].squeeze()
        else:
            logits = torch.tensor(out).squeeze()
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
        best_idx = int(np.argmax(probs))
        best_prob = float(probs[best_idx])
        best_label = labels[best_idx] if labels and best_idx < len(labels) else f"idx_{best_idx}"
        print("\n🔧 Debug: raw classifier output (for troubleshooting):")
        print(out)
        print("\n🔎 Top predictions:")
        topk_idx = np.argsort(probs)[-8:][::-1]
        for i in topk_idx[:8]:
            nm = labels[i] if labels and i < len(labels) else f"idx_{i}"
            print(f"  {nm} — {probs[i]*100:.2f}%")
        print(f"\n🎯 Final Guess: {best_label} — {best_prob*100:.2f}%")
    except Exception as e:
        print("❌ Classification failed. Traceback:")
        traceback.print_exc()
        sys.exit(1)
