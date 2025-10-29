# hf_download_model.py
from huggingface_hub import hf_hub_download
import os, shutil

repo_id = "Jzuluaga/accent-id-commonaccent_ecapa"
dest = os.path.join(os.getcwd(), "pretrained_models", "accent-id-commonaccent_ecapa")
os.makedirs(dest, exist_ok=True)

files = ["hyperparams.yaml", "classifier.ckpt", "accent_encoder.txt", "embedding_model.ckpt"]
for fname in files:
    try:
        p = hf_hub_download(repo_id=repo_id, filename=fname)
        shutil.copy(p, os.path.join(dest, fname))
        print("Downloaded", fname)
    except Exception as e:
        print("Failed to download", fname, e)
print("Done — check", dest)
