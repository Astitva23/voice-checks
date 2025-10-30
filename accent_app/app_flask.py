# app_flask.py (robust, logs details and serves index reliably)
import os, sys, traceback, tempfile, subprocess, shutil
from pathlib import Path
from flask import Flask, request, jsonify, send_file, abort

# -------------------------
# Config - edit if needed
# -------------------------
BASE_DIR = Path(__file__).parent.resolve()
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"
MODEL_DIR = r"E:\new-voice\voice-checks\accent_app\pretrained_models\accent-id-commonaccent_ecapa"

# -------------------------
# Import model helpers (try accent_full_local first)
# -------------------------
try:
    # accent_full_local should export: load_labels_from_model_dir, classify_audio_robust, EncoderClassifier
    from accent_full_local import load_labels_from_model_dir, classify_audio_robust, EncoderClassifier
    print("Imported helpers from accent_full_local.py")
except Exception as e:
    print("Could not import accent_full_local helpers:", e)
    # Minimal fallback stubs (will attempt to import speechbrain when needed)
    def load_labels_from_model_dir(model_dir, classifier_obj=None):
        p = Path(model_dir) / "label_encoder.cleaned.txt"
        if not p.exists():
            p = Path(model_dir) / "label_encoder.txt"
        if p.exists():
            return [ln.strip() for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip() and "starting_index" not in ln.lower()]
        return []

    def classify_audio_robust(classifier, filepath, target_sr=16000):
        # lazy import to reduce startup failures
        from speechbrain.inference import EncoderClassifier as _Enc
        if classifier is None:
            classifier = _Enc.from_hparams(source=MODEL_DIR, savedir=MODEL_DIR)
        return classifier.classify_file(filepath)

    class EncoderClassifier:
        # wrapper to avoid NameError if imported elsewhere — not used here
        pass

# -------------------------
# Start / load classifier (lazy)
# -------------------------
classifier = None
labels = None

def ensure_model_loaded():
    global classifier, labels
    if classifier is not None:
        return
    try:
        print("Loading classifier from:", MODEL_DIR)
        from speechbrain.inference import EncoderClassifier as SBEnc
        classifier = SBEnc.from_hparams(source=MODEL_DIR, savedir=MODEL_DIR)
        labels = load_labels_from_model_dir(MODEL_DIR, classifier_obj=classifier) or []
        print("Loaded classifier. Labels count:", len(labels))
    except Exception:
        traceback.print_exc()
        raise

# -------------------------
# Flask app
# -------------------------
app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

@app.route("/health")
def health():
    return jsonify({"status": "ok", "base_dir": str(BASE_DIR), "static": str(STATIC_DIR), "index_exists": INDEX_FILE.exists()})

@app.route("/")
def index():
    if INDEX_FILE.exists():
        # serve index.html directly (absolute path) to avoid path issues
        return send_file(str(INDEX_FILE))
    else:
        return (
            "<h2>Index file not found</h2>"
            f"<p>Expected at: {INDEX_FILE}</p>"
            "<p>Place your static/index.html file there.</p>", 404
        )

def convert_with_ffmpeg(src_path, dst_path, sr=16000, channels=1):
    ff = shutil.which("ffmpeg")
    if ff is None:
        return False, "ffmpeg not found in PATH"
    cmd = [ff, "-y", "-i", str(src_path), "-ar", str(sr), "-ac", str(channels), "-vn", str(dst_path)]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return True, out.decode("utf-8", errors="ignore")
    except subprocess.CalledProcessError as e:
        stderr = e.output.decode("utf-8", errors="ignore") if e.output is not None else str(e)
        return False, stderr
    except Exception as e:
        return False, str(e)

@app.route("/classify", methods=["POST"])
def classify_route():
    try:
        # lazy load model to avoid import-time failures
        ensure_model_loaded()
        if "file" not in request.files:
            return jsonify({"error": "No file part in request"}), 400
        f = request.files["file"]
        if f.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        tmp_in = Path(tempfile.gettempdir()) / f"uploaded_{os.getpid()}_{int.from_bytes(os.urandom(4), 'little')}.bin"
        f.save(str(tmp_in))

        # Try reading with soundfile first
        import soundfile as sf
        wav_path = None
        try:
            sf.info(str(tmp_in))
            wav_path = str(tmp_in)
        except Exception as e_sf:
            # convert using ffmpeg to WAV (16000 mono)
            converted = Path(tempfile.gettempdir()) / f"converted_{os.getpid()}.wav"
            ok, info = convert_with_ffmpeg(tmp_in, converted)
            if not ok:
                tmp_in.unlink(missing_ok=True)
                return jsonify({"error": "ffmpeg conversion failed", "details": info}), 500
            wav_path = str(converted)

        # classify
        out = classify_audio_robust(classifier, wav_path)
        # parse logits -> probs
        import torch, numpy as np
        if isinstance(out, (list, tuple)) and len(out) >= 1:
            logits = out[0].squeeze()
        else:
            logits = torch.tensor(out).squeeze()
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
        best_idx = int(np.argmax(probs))
        best_conf = float(probs[best_idx])
        best_label = labels[best_idx] if labels and best_idx < len(labels) else f"idx_{best_idx}"
        topk_idx = np.argsort(probs)[-5:][::-1]
        top5 = [{"label": labels[i] if labels and i < len(labels) else str(i), "prob": float(probs[i])} for i in topk_idx]

        # cleanup
        tmp_in.unlink(missing_ok=True)
        if 'converted' in locals() and converted.exists():
            converted.unlink(missing_ok=True)

        return jsonify({"label": best_label, "confidence": best_conf, "top5": top5})
    except Exception as e:
        tb = traceback.format_exc()
        print("Exception in /classify:", tb)
        return jsonify({"error": str(e), "trace": tb}), 500

if __name__ == "__main__":
    print("STARTING app_flask.py")
    print("BASE_DIR:", BASE_DIR)
    print("STATIC_DIR:", STATIC_DIR)
    print("INDEX_FILE exists:", INDEX_FILE.exists())
    print("MODEL_DIR:", MODEL_DIR)
    # launch on 0.0.0.0 so other devices on LAN can hit it if needed
    app.run(host="0.0.0.0", port=5000)
