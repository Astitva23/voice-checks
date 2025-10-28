import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForAudioClassification
import gradio as gr

# Load model
MODEL = "emrysj/AccentClassifier"
processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForAudioClassification.from_pretrained(MODEL)

# Classify accent
def detect_accent(audio):
    if audio is None:
        return "No audio provided."
    waveform, sr = torchaudio.load(audio)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.nn.functional.softmax(logits, dim=-1)
    top_idx = torch.argmax(pred, dim=-1).item()
    score = pred[0, top_idx].item() * 100
    label = model.config.id2label[top_idx]
    return f"{label} ({score:.2f}% confidence)"

# UI
ui = gr.Interface(
    fn=detect_accent,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath", label="Record or Upload Speech"),
    outputs=gr.Textbox(label="Detected Accent"),
    title="Accent Detector",
    description="Upload or record a voice sample to identify the English accent."
)

if __name__ == "__main__":
    ui.launch()
