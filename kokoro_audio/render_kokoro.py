import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

ROOT = Path(__file__).parent
parts = sorted(ROOT.glob("flashcards*.txt"))
TXT = "\n\n".join(p.read_text(encoding="utf-8") for p in parts)
pairs = re.findall(r"Question\s+\d+:\s*(.*?)\nAnswer:\s*(.*?)(?=\n\nQuestion\s+\d+:|\Z)", TXT, re.S)
if len(pairs) != 66:
    raise RuntimeError(f"Expected 66 cards, found {len(pairs)}")

kokoro = Kokoro(str(ROOT / "kokoro-v1.0.onnx"), str(ROOT / "voices-v1.0.bin"))
VOICE = "af_heart"
LANG = "en-us"
SPEED = 1.0
QUESTION_PAUSE_S = 3.0
ANSWER_PAUSE_S = 1.2

segments = []
sr_ref = None
for i, (q, a) in enumerate(pairs, 1):
    q = re.sub(r"\s+", " ", q).strip()
    a = re.sub(r"\s+", " ", a).strip()
    q_audio, sr = kokoro.create(q, voice=VOICE, speed=SPEED, lang=LANG)
    a_audio, sr2 = kokoro.create(a, voice=VOICE, speed=SPEED, lang=LANG)
    if sr != sr2:
        raise RuntimeError("Sample-rate mismatch")
    sr_ref = sr_ref or sr
    if sr_ref != sr:
        raise RuntimeError("Sample-rate changed")
    segments.extend([
        np.asarray(q_audio, dtype=np.float32),
        np.zeros(int(sr * QUESTION_PAUSE_S), dtype=np.float32),
        np.asarray(a_audio, dtype=np.float32),
        np.zeros(int(sr * ANSWER_PAUSE_S), dtype=np.float32),
    ])
    print(f"Rendered {i}/66")

wave = np.concatenate(segments)
peak = float(np.max(np.abs(wave))) or 1.0
if peak > 0.98:
    wave *= 0.98 / peak

wav_path = ROOT / "Data_Organization_Kernel_af_heart.wav"
mp3_path = ROOT / "Data_Organization_Kernel_af_heart.mp3"
sf.write(wav_path, wave, sr_ref, subtype="PCM_24")
subprocess.run([
    "ffmpeg", "-y", "-i", str(wav_path),
    "-af", "loudnorm=I=-16:TP=-1.5:LRA=7",
    "-codec:a", "libmp3lame", "-b:a", "192k", str(mp3_path)
], check=True)
print(mp3_path)
