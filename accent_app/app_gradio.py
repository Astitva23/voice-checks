# app_gradio.py
import os
from pathlib import Path
import gradio as gr
import numpy as np
import torch
import soundfile as sf
from accent_full_local import MODEL_DIR, load_labels_from_model_dir, classify_audio_robust, EncoderClassifier  # see notes

# If accent_full_local has top-level model load code, modify to expose helper functions.
# For simplicity, we'll load our classifier here (no-symlink env var assumed set if needed).
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

MODEL_DIR = MODEL_DIR  # from accent_full_local module
print("Loading model for Gradio UI...")
classifier = EncoderClassifier.from_hparams(source=MODEL_DIR, savedir=MODEL_DIR)
labels = load_labels_from_model_dir(MODEL_DIR, classifier_obj=classifier)

def classify_file_from_upload(wav):
    """
    wav: either a (sr, np.array) tuple from gradio mic or path to file
    gradio microphone returns tuple (sr, audio_np) when passed to the function directly
    """
    # handle (sr, array)
    if isinstance(wav, tuple) or isinstance(wav, list):
        sr, arr = wav[0], np.array(wav[1])
        # ensure mono
        if arr.ndim > 1:
            arr = np.mean(arr, axis=1)
        # write to temporary file
        tmp = Path("tmp_uploaded.wav")
        sf.write(str(tmp), arr.astype(np.float32), sr, subtype="PCM_16")
        path = str(tmp)
    elif isinstance(wav, str) and os.path.exists(wav):
        path = wav
    else:
        return "Invalid input", None

    # use classify_audio_robust to get output
    try:
        out = classify_audio_robust(classifier, path)
    except Exception as e:
        return f"Error during classification: {e}", None

    # parse logits -> probs
    if isinstance(out, (list, tuple)) and len(out) >= 1:
        logits = out[0].squeeze()
    else:
        logits = torch.tensor(out).squeeze()
    probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
    best_idx = int(np.argmax(probs))
    best_conf = probs[best_idx]
    best_label = labels[best_idx] if labels and best_idx < len(labels) else f"idx_{best_idx}"
    topk_idx = np.argsort(probs)[-5:][::-1]
    topk = [(labels[i] if labels and i < len(labels) else f"idx_{i}", float(probs[i])) for i in topk_idx]
    # format output
    out_text = f"{best_label} — {best_conf*100:.2f}%"
    topk_text = "\n".join([f"{n}: {p*100:.2f}%" for n,p in topk])
    return out_text, topk_text

with gr.Blocks() as demo:
    gr.Markdown("# Accent Classifier")
    with gr.Row():
        mic = gr.Audio(source="microphone", type="numpy", label="Record (microphone)")
        upload = gr.Audio(source="upload", type="filepath", label="Or upload a WAV")
    classify_btn = gr.Button("Classify")
    result_label = gr.Textbox(label="Top prediction", interactive=False)
    top5 = gr.Textbox(label="Top 5 (name: confidence)", interactive=False)
    def classify_click(mic_val, upload_val):
        # prefer upload file if provided, else mic
        wav_input = None
        if upload_val:
            wav_input = upload_val
        elif mic_val:
            wav_input = mic_val
        else:
            return "No audio provided", ""
        return classify_file_from_upload(wav_input)

    classify_btn.click(classify_click, inputs=[mic, upload], outputs=[result_label, top5])

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
