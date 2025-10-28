# accent_speechbrain.py
import os, sys
import soundfile as sf
import sounddevice as sd
from speechbrain.pretrained import EncoderClassifier

INPUT_FILE = "./voice.wav"
RECORD_SECONDS = 10
MODEL = "Jzuluaga/accent-id-commonaccent_ecapa"  # public SpeechBrain model

def record_if_missing(path, dur=10, sr=16000):
    if os.path.exists(path):
        return
    print(f"Recording {dur}s to {path} (speak naturally)...")
    rec = sd.rec(int(dur * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    sf.write(path, rec, sr)
    print("Saved recording.")

def main():
    record_if_missing(INPUT_FILE, RECORD_SECONDS)

    print("Loading SpeechBrain model (may download on first run)...")
    try:
        classifier = EncoderClassifier.from_hparams(
    source="pretrained_models/accent-id-commonaccent_ecapa",
    savedir="pretrained_models/accent-id-commonaccent_ecapa"
    )

    except Exception as e:
        print("Model load failed:", e)
        sys.exit(1)
    print("Model loaded.\nClassifying...")

    # classify_file returns (logits, score, index, labels)
    out = classifier.classify_file(INPUT_FILE)
    # SpeechBrain's custom interface returns out as tuples; try to parse robustly
    try:
        logits = out[0].squeeze()
        labels = out[3]
    except Exception:
        print("Unexpected output format from classifier:", type(out))
        print(out)
        sys.exit(1)

    import torch, numpy as np
    probs = torch.softmax(logits, dim=0).cpu().numpy()
    topk = np.argsort(probs)[-5:][::-1]

    print("\nTop predictions:")
    for i in topk:
        print(f"  {labels[i]} — {probs[i]*100:.2f}%")

    # print final top-1
    idx = int(np.argmax(probs))
    print(f"\nFinal guess: {labels[idx]} — {probs[idx]*100:.2f}%")

if __name__ == "__main__":
    main()
