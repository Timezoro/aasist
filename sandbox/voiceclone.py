"""
Voice clone robustness test for AASIST.

Runs the pretrained AASIST model against:
  - sandbox/enroll.wav       (real voice)
  - three ElevenLabs MP3s    (synthetic clones)

Prints per-file scores and a summary verdict.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.AASIST import Model

# ── constants ────────────────────────────────────────────────────────────────
NB_SAMP = 64600          # ~4 s at 16 kHz
TARGET_SR = 16_000
CONFIG_PATH = ROOT / "config" / "AASIST.conf"
WEIGHTS_PATH = ROOT / "models" / "weights" / "AASIST.pth"
SANDBOX = Path("E:/Github/voice_clone")

REAL_FILES = [
    ("9Arm.wav",      "real"),
    ("Me.wav",        "real"),
    ("Streamer.wav",  "real"),
    ("Top.wav",       "real"),
    ("Youtuber.wav",  "real"),
]
FAKE_FILES = [
    ("9Arm_clone.mp3",                 "fake"),
    ("MeIntroduce_Clone.mp3",          "fake"),
    ("MeIntorducebutgirl_clone.mp3",   "fake"),
    ("Streamer_clone.mp3",             "fake"),
    ("Top_clone.mp3",                  "fake"),
]
ALL_FILES = REAL_FILES + FAKE_FILES


# ── audio loading ─────────────────────────────────────────────────────────────
def load_audio(path: Path) -> np.ndarray:
    """Load audio to mono 16 kHz numpy array, pad/trim to NB_SAMP."""
    suffix = path.suffix.lower()
    if suffix == ".wav":
        import soundfile as sf
        x, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
    elif suffix in (".mp3", ".m4a", ".ogg", ".flac"):
        import torchaudio
        waveform, sr = torchaudio.load(str(path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
            waveform = resampler(waveform)
        x = waveform.numpy()
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    # resample WAV if needed
    if suffix == ".wav" and sr != TARGET_SR:
        import torchaudio
        t = torch.from_numpy(x).unsqueeze(0)
        t = torchaudio.transforms.Resample(sr, TARGET_SR)(t)
        x = t.squeeze(0).numpy()

    # pad or trim to NB_SAMP
    if len(x) >= NB_SAMP:
        x = x[:NB_SAMP]
    else:
        reps = int(NB_SAMP / len(x)) + 1
        x = np.tile(x, reps)[:NB_SAMP]
    return x.astype(np.float32)


# ── model loading ─────────────────────────────────────────────────────────────
def load_model(device: torch.device) -> Model:
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    model = Model(config["model_config"])
    state = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ── inference ─────────────────────────────────────────────────────────────────
def score_file(model: Model, path: Path, device: torch.device) -> dict:
    """Return softmax probs and verdict for one file."""
    x = load_audio(path)
    t = torch.from_numpy(x).unsqueeze(0).to(device)   # (1, NB_SAMP)
    with torch.no_grad():
        _, logits = model(t)                            # (1, 2)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu()
    spoof_prob    = probs[0].item()
    bonafide_prob = probs[1].item()
    verdict = "BONAFIDE" if bonafide_prob > 0.5 else "SPOOF"
    return {
        "spoof_prob":    spoof_prob,
        "bonafide_prob": bonafide_prob,
        "verdict":       verdict,
    }


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Loading model from {WEIGHTS_PATH} ...")
    model = load_model(device)
    print("Model loaded.\n")

    results = []
    header = f"{'File':<60} {'Label':<6} {'Bonafide%':>10} {'Spoof%':>8}  Verdict"
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for fname, label in ALL_FILES:
        path = SANDBOX / fname
        res = score_file(model, path, device)
        correct = (label == "real" and res["verdict"] == "BONAFIDE") or \
                  (label == "fake" and res["verdict"] == "SPOOF")
        tag = "[OK]" if correct else "[MISS]"
        short = (fname[:57] + "...") if len(fname) > 60 else fname
        print(f"{short:<60} {label:<6} {res['bonafide_prob']*100:>9.2f}%"
              f" {res['spoof_prob']*100:>7.2f}%  {res['verdict']} {tag}")
        results.append({"file": fname, "label": label, **res})

    print(sep)

    correct_count = sum(
        1 for r in results
        if (r["label"] == "real" and r["verdict"] == "BONAFIDE") or
           (r["label"] == "fake" and r["verdict"] == "SPOOF")
    )
    total = len(results)
    print(f"\nAccuracy: {correct_count}/{total}  ({correct_count/total*100:.0f}%)")

    real_res = [r for r in results if r["label"] == "real"]
    fake_res = [r for r in results if r["label"] == "fake"]
    avg_real = np.mean([r["bonafide_prob"] for r in real_res])
    avg_fake = np.mean([r["bonafide_prob"] for r in fake_res])
    print(f"Avg bonafide confidence  — real: {avg_real*100:.1f}%   fake: {avg_fake*100:.1f}%")

    return results


if __name__ == "__main__":
    results = main()
