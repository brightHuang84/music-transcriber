"""Major/minor chord labels from the chroma of the pitched instruments.

These labels are a learning aid, not a lead sheet. Ambiguous frames are left blank.
"""

from __future__ import annotations

import numpy as np

from app.audio_io import to_mono

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler key profiles.
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def chord_display_name(symbol: str) -> str:
    if symbol.endswith("m") and not symbol.endswith("dim"):
        return f"{symbol[:-1]} 小和弦"
    return f"{symbol} 大和弦"


def _templates() -> list[tuple[str, np.ndarray]]:
    major = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float)
    minor = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float)
    items: list[tuple[str, np.ndarray]] = []
    for shift, name in enumerate(PITCH_NAMES):
        items.append((name, np.roll(major, shift)))
        items.append((name + "m", np.roll(minor, shift)))
    return items


def _match_chord(chroma: np.ndarray) -> tuple[str | None, float]:
    vector = np.asarray(chroma, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < 1e-6:
        return None, 0.0
    vector = vector / norm
    scored: list[tuple[float, str]] = []
    for name, template in _templates():
        template = template / (np.linalg.norm(template) + 1e-8)
        scored.append((float(np.dot(vector, template)), name))
    scored.sort(reverse=True)
    best_score, best_name = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    if best_score < 0.55 or best_score - second < 0.04:
        return None, best_score
    return best_name, best_score


def _merge(chords: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for chord in chords:
        if merged and merged[-1]["symbol"] == chord["symbol"] and chord["start"] <= merged[-1]["end"] + 0.05:
            merged[-1]["end"] = chord["end"]
        else:
            merged.append(dict(chord))
    return merged


def detect_key(chroma: np.ndarray) -> dict | None:
    mean = np.mean(chroma, axis=1)
    if float(np.sum(mean)) < 1e-6:
        return None
    best_name = None
    best_score = -1e9
    for shift, name in enumerate(PITCH_NAMES):
        for mode, profile, label in (
            ("major", _MAJOR_PROFILE, "大调"),
            ("minor", _MINOR_PROFILE, "小调"),
        ):
            score = float(np.corrcoef(mean, np.roll(profile, shift))[0, 1])
            if np.isnan(score):
                continue
            if score > best_score:
                best_score = score
                best_name = {"tonic": name, "mode": mode, "name": f"{name} {label}", "confidence": round(score, 3)}
    return best_name


def detect_chords(audio: np.ndarray, sample_rate: int, beats: list[float], duration: float) -> dict:
    import librosa

    mono = to_mono(audio)
    if mono.size < sample_rate or float(np.sqrt(np.mean(mono**2))) < 1e-4:
        return {"chords": [], "key": None}
    harmonic = librosa.effects.harmonic(mono, margin=2.0)
    hop = 2048
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sample_rate, hop_length=hop)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sample_rate, hop_length=hop)

    boundaries = [0.0]
    boundaries.extend(beats)
    if not boundaries or boundaries[-1] < duration:
        boundaries.append(duration)
    # Fall back to one-second slices when beat tracking found almost nothing.
    if len(boundaries) < 3:
        boundaries = list(np.arange(0, duration, 1.0)) + [duration]

    raw: list[dict] = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end - start < 0.05:
            continue
        chosen = (times >= start) & (times < end)
        if not np.any(chosen):
            continue
        symbol, score = _match_chord(np.mean(chroma[:, chosen], axis=1))
        if symbol is None:
            continue
        raw.append(
            {
                "start": round(float(start), 4),
                "end": round(float(end), 4),
                "symbol": symbol,
                "name": chord_display_name(symbol),
                "confidence": round(score, 3),
            }
        )
    return {"chords": _merge(raw), "key": detect_key(chroma)}
