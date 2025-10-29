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
INPUT_FILE = r"E:\new-voice\voice-checks\accent_app\test_audio"

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
print("✅ Model loaded. Classifying:", INPUT_FILE)
try:
    out = classifier.classify_file(INPUT_FILE)
except Exception as e:
    print("❌ classify_file failed. Traceback:")
    traceback.print_exc()
    sys.exit(1)

# ======== Parse and print results robustly ========
try:
    import torch, numpy as np
    logits = out[0].squeeze()
    labels = list(out[3])
    probs = torch.softmax(logits, dim=0).cpu().numpy()
    topk = np.argsort(probs)[-5:][::-1]
    print("\n🔎 Top predictions:")
    for i in topk:
        print(f"  {labels[i]} — {probs[i]*100:.2f}%")
    idx = int(np.argmax(probs))
    print(f"\n🎯 Final guess: {labels[idx]} — {probs[idx]*100:.2f}%")
except Exception:
    print("⚠️ Unexpected output format from classifier. Raw output:")
    print(out)
    traceback.print_exc()
    sys.exit(1)
