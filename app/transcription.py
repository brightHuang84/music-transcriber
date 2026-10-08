"""Transcribe a pitched stem to notes with Spotify Basic Pitch (ONNX)."""

from __future__ import annotations

import contextlib
import io
import threading
from pathlib import Path

import numpy as np

from app.errors import UserFacingError
from app.notes import describe_pitch
from app.stems import label_for

# Per-stem ranges keep bass out of the melody octave and vocals out of sub-bass rumble.
_STEM_SETTINGS = {
    "vocals": {
        "minimum_frequency": 75.0,
        "maximum_frequency": 1800.0,
        "melodia_trick": True,
        "minimum_note_length": 90.0,
    },
    "bass": {
        "minimum_frequency": 28.0,
        "maximum_frequency": 450.0,
        "melodia_trick": True,
        "minimum_note_length": 80.0,
    },
    "guitar": {
        "minimum_frequency": 70.0,
        "maximum_frequency": 1400.0,
        "melodia_trick": False,
        "minimum_note_length": 70.0,
    },
    "piano": {
        "minimum_frequency": 27.0,
        "maximum_frequency": 4200.0,
        "melodia_trick": False,
        "minimum_note_length": 60.0,
    },
    "other": {
        "minimum_frequency": 50.0,
        "maximum_frequency": 4200.0,
        "melodia_trick": False,
        "minimum_note_length": 80.0,
    },
}

_model = None
_model_lock = threading.Lock()


def _onnx_model_path() -> Path:
    import basic_pitch

    bundled = Path(basic_pitch.__file__).resolve().parent / "saved_models" / "icassp_2022" / "nmp.onnx"
    if not bundled.exists():
        raise UserFacingError("音符识别模型文件缺失。请重新运行 start 脚本，让它装好 basic-pitch。")
    return bundled


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            try:
                from basic_pitch.inference import Model

                _model = Model(_onnx_model_path())
            except Exception as exc:
                raise UserFacingError("音符识别模型加载失败。请重新运行 start 脚本后再试。") from exc
        return _model


def _merge_same_pitch(notes: list[dict]) -> list[dict]:
    ordered = sorted(notes, key=lambda note: (note["pitch"], note["start"]))
    merged: list[dict] = []
    for note in ordered:
        if (
            merged
            and merged[-1]["pitch"] == note["pitch"]
            and note["start"] <= merged[-1]["end"] + 0.03
        ):
            merged[-1]["end"] = max(merged[-1]["end"], note["end"])
            merged[-1]["velocity"] = max(merged[-1]["velocity"], note["velocity"])
        else:
            merged.append(dict(note))
    merged.sort(key=lambda note: (note["start"], note["pitch"]))
    for note in merged:
        note["start"] = round(float(note["start"]), 4)
        note["end"] = round(float(note["end"]), 4)
        note["duration"] = round(note["end"] - note["start"], 4)
    return merged


def _drop_weak_octave_duplicates(notes: list[dict]) -> list[dict]:
    """Drop a quiet note one octave above a stronger note sounding at the same time.

    Harmonic-rich tones often produce that extra octave. A real octave that is
    almost as loud is kept.
    """
    ordered = sorted(notes, key=lambda note: (note["start"], -note["velocity"]))
    kept: list[dict] = []
    for note in ordered:
        weaker_octave = False
        for other in kept:
            overlap = min(note["end"], other["end"]) - max(note["start"], other["start"])
            if overlap <= 0.06:
                continue
            if note["pitch"] == other["pitch"] + 12 and note["velocity"] < other["velocity"] * 0.75:
                weaker_octave = True
                break
        if not weaker_octave:
            kept.append(note)
    kept.sort(key=lambda note: (note["start"], note["pitch"]))
    return kept


def _drop_inaudible(notes: list[dict], audio_path: Path, floor: float = 0.005) -> list[dict]:
    """Ignore notes Basic Pitch draws on a stretch that is effectively silent.

    An empty stem still has a noise floor. If a later section is loud enough
    that the whole file is not skipped, those quiet stretches get a fake melody.
    """
    from app.audio_io import load_audio, to_mono

    audio, sample_rate = load_audio(audio_path)
    mono = to_mono(audio)
    kept: list[dict] = []
    for note in notes:
        start = max(0, int(float(note["start"]) * sample_rate))
        end = int(max(float(note["end"]), float(note["start"]) + 0.05) * sample_rate)
        segment = mono[start:end]
        if segment.size == 0:
            continue
        level = float(np.sqrt(np.mean(np.square(segment.astype(np.float64)))))
        if level < floor:
            continue
        kept.append(note)
    return kept


def transcribe(audio_path: Path, stem: str, midi_tempo: float = 120.0) -> list[dict]:
    """Return notes with pitch, onset, duration, and velocity."""
    from basic_pitch.inference import predict

    settings = _STEM_SETTINGS.get(stem, _STEM_SETTINGS["other"])
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            _model_output, _midi, note_events = predict(
                str(audio_path),
                get_model(),
                onset_threshold=0.5,
                frame_threshold=0.3,
                minimum_note_length=settings["minimum_note_length"],
                minimum_frequency=settings["minimum_frequency"],
                maximum_frequency=settings["maximum_frequency"],
                melodia_trick=settings["melodia_trick"],
                midi_tempo=float(midi_tempo) if midi_tempo and midi_tempo > 0 else 120.0,
            )
    except UserFacingError:
        raise
    except Exception as exc:
        raise UserFacingError(f"识别{label_for(stem)}的音高时出了问题。可以换一首更短、更清晰的歌再试。") from exc

    notes: list[dict] = []
    for start, end, pitch, amplitude, _bends in note_events:
        duration = float(end) - float(start)
        if duration < 0.05:
            continue
        described = describe_pitch(int(pitch))
        velocity = int(np.clip(round(float(amplitude) * 127), 1, 127))
        notes.append(
            {
                **described,
                "start": float(start),
                "end": float(end),
                "duration": duration,
                "velocity": velocity,
            }
        )
    return _drop_inaudible(_drop_weak_octave_duplicates(_merge_same_pitch(notes)), audio_path)
