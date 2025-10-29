# inspect_labels.py
import os
MODEL_DIR = r"E:\new-voice\voice-checks\accent_app\pretrained_models\accent-id-commonaccent_ecapa"
print("Model dir:", MODEL_DIR)
for root,dirs,files in os.walk(MODEL_DIR):
    for f in files:
        print("-", os.path.join(root, f))
# Try to print likely label files
candidates = [
    os.path.join(MODEL_DIR, "label_encoder.txt"),
    os.path.join(MODEL_DIR, "accent_encoder.txt"),
    os.path.join(MODEL_DIR, "data", "label_encoder.txt")
]
for p in candidates:
    if os.path.exists(p):
        print("\n=== contents of", p, "===\n")
        print(open(p, "r", encoding="utf-8", errors="replace").read())
