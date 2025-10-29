#!/usr/bin/env python3
"""
accent_speechbrain_local.py

- Loads a SpeechBrain accent model from the local folder:
    pretrained_models/accent-id-commonaccent_ecapa/
- Reads a test wav at: test_audio/voice.wav
- Does NOT record or require internet.
- Prints top-5 predictions or clear error messages.

Place your files as:
project/
  ├─ accent_speechbrain_local.py
  ├─ pretrained_models/
  │    └─ accent-id-commonaccent_ecapa/  <-- model files here
  └─ test_audio/
       └─ voice.wav
"""

import os, sys
import soundfile as sf
import numpy as np

# Path config (edit if needed)
INPUT_FILE = os.path.join("test_audio", "voice.wav")
LOCAL_MODEL_DIR = os.path.join("pretrained_models", "accent-id-commonaccent_ecapa")

# Helpful function: friendly exit
def die(msg):
    print("\n❌", msg)
    sys.exit(1)

# 1) Basic checks: script run from project root?
if not os.path.exists(os.getcwd()):
    die("Current working directory not found. Run this script from your project folder.")

print("🔎 Checking required files and folders...")

# 2) Check test audio
if not os.path.exists(INPUT_FILE):
    die(f"Input audio not found at '{INPUT_FILE}'.\nPlease put your pre-recorded file named 'voice.wav' into the 'test_audio' folder (create it if needed).")

# Check audio can be read
try:
    y, sr = sf.read(INPUT_FILE, always_2d=False)
except Exception as e:
    die(f"Failed to read '{INPUT_FILE}': {e}\nCheck that the file is a valid WAV and not corrupted. Try converting with ffmpeg: ffmpeg -i input.mp3 -ar 16000 -ac 1 test_audio/voice.wav")

print(f"✅ Test audio found: {INPUT_FILE} (samplerate {sr}, duration {len(y)/sr:.2f}s)")

# 3) Check local model folder
if not os.path.isdir(LOCAL_MODEL_DIR):
    die(f"Local model directory not found at '{LOCAL_MODEL_DIR}'.\nIf you already downloaded the model, ensure folder name and path are correct. Otherwise: download the model files and place them under this path.")

# Look for expected files (speechbrain usually needs hyperparams.yaml, classifier.ckpt, label_encoder.txt etc.)
expected_files = ["hyperparams.yaml", "classifier.ckpt", "label_encoder.txt"]
found = {f: os.path.exists(os.path.join(LOCAL_MODEL_DIR, f)) for f in expected_files}
missing = [f for f, ok in found.items() if not ok]
if missing:
    print("⚠️ Some expected model files are missing:", missing)
    print("Model folder contents:", os.listdir(LOCAL_MODEL_DIR))
    die("Ensure you copied the full model folder (all files) into pretrained_models/accent-id-commonaccent_ecapa")

# 4) Load SpeechBrain locally (robust import + error messages)
try:
    # new recommended import
    from speechbrain.inference import EncoderClassifier

except Exception as e:
    die(f"Failed to import SpeechBrain. Error: {e}\nMake sure speechbrain is installed in this environment: pip install speechbrain")

print("🧠 Loading the SpeechBrain model from local folder (this will not download anything)...")
try:
    classifier = EncoderClassifier.classifier = EncoderClassifier.from_hparams(
    source="pretrained_models/accent-id-commonaccent_ecapa",
    savedir="pretrained_models/accent-id-commonaccent_ecapa"
)

except Exception as e:
    die(f"Model load failed: {e}\nIf you see a Windows permission or symlink error, try:\n  - Move the project out of OneDrive to a short path like C:\\projects\\voice-modulation\n  - Set environment variables: setx HF_HUB_DISABLE_SYMLINKS 1  and  setx HF_HUB_DISABLE_SYMLINKS_WARNING 1\n  - Or run the script from an Administrator terminal once.\nOr, alternatively, re-download the model files manually and ensure the folder structure is correct.")

print("✅ Model loaded successfully.\nClassifying...")

# 5) Classify file and print results robustly
try:
    out = classifier.classify_file(INPUT_FILE)
except Exception as e:
    die(f"Model classification failed: {e}\nCheck that the model supports .classify_file and the input audio is not empty/corrupted.")

# Parse output safely
try:
    logits = out[0].squeeze()
    labels = out[3]
    import torch
    probs = torch.softmax(logits, dim=0).cpu().numpy()
except Exception as e:
    die(f"Unexpected model output format: {e}\nRaw output: {type(out)}\n{out}")

# Show top-5
topk_idx = np.argsort(probs)[-5:][::-1]
print("\n🔎 Top predictions:")
for i in topk_idx:
    print(f"  {labels[i]} — {probs[i]*100:.2f}%")

idx = int(np.argmax(probs))
print(f"\n🎯 Final guess: {labels[idx]} — {probs[idx]*100:.2f}%")
print("\nDone.")
