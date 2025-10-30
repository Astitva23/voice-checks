"""
run_accent_local_nosymlink.py
--------------------------------
Runs accent classification using a local SpeechBrain model
without creating any symlinks (fixes WinError 1314).
Edit the paths below if you move the model or audio later.
"""

import os
import sys
import traceback
import shutil
import torch
import numpy as np
from speechbrain.inference import EncoderClassifier

# =====================================================
# 🔧 USER SETTINGS (EDIT THESE TWO PATHS AS NEEDED)
# =====================================================
MODEL_DIR = r"E:\new-voice\voice-checks\accent_app\pretrained_models\accent-id-commonaccent_ecapa"
INPUT_FILE = r"E:\new-voice\voice-checks\accent_app\test_audio\voice.wav"
# =====================================================

# Force Hugging Face to never use symlinks (Windows-safe)
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Use a project-local Hugging Face cache (inside your folder)
project_cache = os.path.join(os.path.dirname(__file__), ".hf_cache")
os.environ["HF_HOME"] = project_cache
os.environ["TRANSFORMERS_CACHE"] = project_cache
os.environ["HF_DATASETS_CACHE"] = project_cache

print("⚙️  Environment variables set:")
print("  HF_HOME =", os.environ["HF_HOME"])
print("  HF_HUB_DISABLE_SYMLINKS =", os.environ["HF_HUB_DISABLE_SYMLINKS"])
print()

# =====================================================
# 🧩 VALIDATE FILE STRUCTURE
# =====================================================
if not os.path.exists(MODEL_DIR):
    print(f"❌ Model folder not found:\n   {MODEL_DIR}")
    sys.exit(1)

if not os.path.exists(INPUT_FILE):
    print(f"❌ Input file not found:\n   {INPUT_FILE}")
    sys.exit(1)

# Ensure label file name consistency
label_file = os.path.join(MODEL_DIR, "label_encoder.txt")
alt_names = [
    os.path.join(MODEL_DIR, "accent_encoder.txt"),
    os.path.join(MODEL_DIR, "accent_encoder"),
]
if not os.path.exists(label_file):
    for alt in alt_names:
        if os.path.exists(alt):
            try:
                shutil.copy2(alt, label_file)
                print(f"ℹ️ Copied {alt} → label_encoder.txt")
                break
            except Exception as e:
                print("⚠️ Could not copy label file:", e)

# =====================================================
# 🧠 LOAD MODEL (NO SYMLINKS)
# =====================================================
print(f"🧠 Loading model from:\n   {MODEL_DIR}\n")
try:
    classifier = EncoderClassifier.from_hparams(source=MODEL_DIR, savedir=MODEL_DIR)
    print("✅ Model loaded successfully!\n")
except Exception:
    print("❌ Model load failed! Full traceback below:\n")
    traceback.print_exc()
    print("\n💡 Tip: Try running this once as Administrator if Windows still blocks access.")
    sys.exit(1)

# =====================================================
# 🎧 CLASSIFY AUDIO FILE
# =====================================================
print(f"🔍 Classifying file:\n   {INPUT_FILE}\n")
try:
    result = classifier.classify_file(INPUT_FILE)
except Exception:
    print("❌ Classification failed! Full traceback below:\n")
    traceback.print_exc()
    sys.exit(1)

# =====================================================
# 📊 SHOW RESULTS
# =====================================================
try:
    logits = result[0].squeeze()
    labels = list(result[3])
    probs = torch.softmax(logits, dim=0).cpu().numpy()

    topk = np.argsort(probs)[-5:][::-1]
    print("🔎 Top 5 Predictions:")
    for i in topk:
        print(f"  {labels[i]} — {probs[i]*100:.2f}%")

    best = int(np.argmax(probs))
    print(f"\n🎯 Final Guess: {labels[best]} — {probs[best]*100:.2f}%")

except Exception:
    print("⚠️ Unexpected output format! Raw output:")
    print(result)
    traceback.print_exc()
