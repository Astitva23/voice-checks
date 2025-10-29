# copy_hf_model_cache_to_local.py
import os, shutil, glob
from pathlib import Path

# Update these if different
USER = "Jzuluaga"
REPO = "accent-id-commonaccent_ecapa"
target_dir = os.path.join(os.getcwd(), "pretrained_models", REPO)

# Find the hf cache snapshot
cache_root = os.path.expanduser(r"~/.cache/huggingface/hub")
pattern = os.path.join(cache_root, f"models--{USER}--{REPO}", "snapshots", "*")
candidates = glob.glob(pattern)
if not candidates:
    # broaden search if necessary
    candidates = glob.glob(os.path.join(cache_root, "models--*", "snapshots", "*"))
    candidates = [p for p in candidates if USER in p and REPO in p]

if not candidates:
    print("❌ No cached snapshot found under ~/.cache/huggingface/hub.")
    print("Please download the model manually from Hugging Face or run a script that pulls the model once.")
    raise SystemExit(1)

# pick newest
candidates.sort(key=os.path.getmtime, reverse=True)
snap = candidates[0]
print("Using snapshot:", snap)

os.makedirs(target_dir, exist_ok=True)

for root, dirs, files in os.walk(snap):
    rel = os.path.relpath(root, snap)
    dest_root = os.path.join(target_dir, rel) if rel != "." else target_dir
    os.makedirs(dest_root, exist_ok=True)
    for f in files:
        src = os.path.join(root, f)
        dst = os.path.join(dest_root, f)
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print("Failed to copy", src, "->", dst, ":", e)

# If model uses accent_encoder.txt -> produce label_encoder.txt
ae = os.path.join(target_dir, "accent_encoder.txt")
le = os.path.join(target_dir, "label_encoder.txt")
if os.path.exists(ae) and not os.path.exists(le):
    shutil.copy2(ae, le)
    print("Copied accent_encoder.txt -> label_encoder.txt")

print("✅ Files copied to:", target_dir)
