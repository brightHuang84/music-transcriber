"""Fine-mode separation of piano, violin, and guitar with known notes.

A plain sine mix is not what Demucs was trained on: low piano falls into the
bass stem, and the rest can collapse into one stem. FluidSynth's General MIDI
piano, violin, and acoustic guitar are close enough to real instruments that
the 6-stem model can be checked. The test skips when that soundfont is absent,
so a normal install is not required to run the rest of the suite.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from app.audio_io import load_audio, to_mono
from app.pipeline import analyze
from app.synth import synthesize_ensemble

pytestmark = pytest.mark.slow

_SOUNDFONTS = (
    Path("/usr/share/sounds/sf2/FluidR3_GM.sf2"),
    Path("/usr/share/sounds/sf2/default-GM.sf2"),
    Path("/usr/share/soundfonts/FluidR3_GM.sf2"),
)


def _soundfont() -> Path | None:
    if shutil.which("fluidsynth") is None:
        return None
    for path in _SOUNDFONTS:
        if path.is_file():
            return path
    return None


def _render_gm(path: Path, clip: dict) -> None:
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(initial_tempo=float(clip["bpm"]))
    for program, key, name in (
        (0, "piano", "Piano"),
        (40, "violin", "Violin"),
        (25, "guitar", "Guitar"),
    ):
        instrument = pretty_midi.Instrument(program=program, name=name)
        for start, end, pitch in clip[key]:
            instrument.notes.append(
                pretty_midi.Note(velocity=90, pitch=int(pitch), start=float(start), end=float(end))
            )
        midi.instruments.append(instrument)
    score = path.with_suffix(".mid")
    midi.write(str(score))
    subprocess.run(
        [
            "fluidsynth",
            "-ni",
            "-g",
            "1.0",
            str(_soundfont()),
            str(score),
            "-F",
            str(path),
            "-r",
            "44100",
        ],
        check=True,
        capture_output=True,
    )


def _solo(path: Path, program: int, notes: list[tuple]) -> np.ndarray:
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    instrument = pretty_midi.Instrument(program=program, name="solo")
    for start, end, pitch in notes:
        instrument.notes.append(
            pretty_midi.Note(velocity=90, pitch=int(pitch), start=float(start), end=float(end))
        )
    midi.instruments.append(instrument)
    score = path.with_suffix(".mid")
    midi.write(str(score))
    wav = path.with_suffix(".wav")
    subprocess.run(
        ["fluidsynth", "-ni", str(_soundfont()), str(score), "-F", str(wav), "-r", "44100"],
        check=True,
        capture_output=True,
    )
    audio, _rate = load_audio(wav)
    return to_mono(audio)


def _recall(expected: list[tuple], detected: list[dict], tolerance: float = 0.45) -> float:
    if not expected:
        return 1.0
    found = 0
    for start, _end, pitch in expected:
        if any(note["pitch"] == pitch and abs(note["start"] - start) <= tolerance for note in detected):
            found += 1
    return found / len(expected)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    count = min(left.size, right.size)
    if count < 8:
        return 0.0
    a = left[:count].astype(np.float64)
    b = right[:count].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def test_fine_mode_separates_gm_piano_violin_and_guitar(tmp_path: Path):
    if _soundfont() is None:
        pytest.skip("FluidSynth and a General MIDI soundfont are not installed")
    clip = synthesize_ensemble()
    source = tmp_path / "ensemble.wav"
    _render_gm(source, clip)
    solos = {
        "piano": _solo(tmp_path / "piano", 0, clip["piano"]),
        "violin": _solo(tmp_path / "violin", 40, clip["violin"]),
        "guitar": _solo(tmp_path / "guitar", 25, clip["guitar"]),
    }
    updates: list[tuple[int, str, str]] = []

    def progress(percent: int, step: str, message: str) -> None:
        updates.append((percent, step, message))

    result = analyze(source, tmp_path / "work", progress, mode="fine")
    assert updates[-1][0] == 100
    assert any(step == "download" for _percent, step, _message in updates)
    assert result["mode"] == "fine"
    assert set(result["stems"]) == {"vocals", "drums", "bass", "guitar", "piano", "other"}

    stems = {}
    for name in result["stems"]:
        audio, _rate = load_audio(tmp_path / "work" / "stems" / f"{name}.wav")
        stems[name] = to_mono(audio)
    scores = {
        instrument: {name: round(_correlation(signal, stems[name]), 3) for name in stems}
        for instrument, signal in solos.items()
    }
    recalls = {
        "piano_in_piano": round(_recall(clip["piano"], result["stems"]["piano"]["notes"]), 3),
        "guitar_in_guitar": round(_recall(clip["guitar"], result["stems"]["guitar"]["notes"]), 3),
        "violin_in_other": round(_recall(clip["violin"], result["stems"]["other"]["notes"]), 3),
        "violin_in_piano": round(_recall(clip["violin"], result["stems"]["piano"]["notes"]), 3),
    }
    print("ISOLATION", scores)
    print("RECALL", recalls)

    def best(instrument: str) -> str:
        return max(scores[instrument], key=scores[instrument].get)

    assert best("piano") == "piano", scores
    assert scores["piano"]["piano"] >= 0.75
    assert best("guitar") == "guitar", scores
    assert scores["guitar"]["guitar"] >= 0.75
    # There is no violin stem. The bowed line should come out in Other.
    assert best("violin") == "other", scores
    assert scores["violin"]["other"] >= 0.75
    assert recalls["piano_in_piano"] >= 0.8, recalls
    assert recalls["guitar_in_guitar"] >= 0.5, recalls
    assert recalls["violin_in_other"] >= 0.75, recalls
    assert recalls["violin_in_piano"] <= 0.25, recalls
    assert not result["stems"]["piano"]["silent"]
    assert not result["stems"]["guitar"]["silent"]
    assert not result["stems"]["other"]["silent"]
    assert result["stems"]["drums"]["silent"]
    assert result["stems"]["bass"]["silent"]
    assert result["stems"]["vocals"]["silent"]

    import pretty_midi

    piano_midi = pretty_midi.PrettyMIDI(str(tmp_path / "work" / "midi" / "piano.mid"))
    guitar_midi = pretty_midi.PrettyMIDI(str(tmp_path / "work" / "midi" / "guitar.mid"))
    combined = pretty_midi.PrettyMIDI(str(tmp_path / "work" / "midi" / "all.mid"))
    assert piano_midi.instruments[0].program == 0
    assert guitar_midi.instruments[0].program == 25
    names = {instrument.name for instrument in combined.instruments}
    # pretty_midi drops an instrument that has no notes, so the silent
    # vocals, bass, and drum tracks are not in the file it reads back.
    assert {"Piano", "Guitar", "Other"} <= names
    assert (tmp_path / "work" / "midi" / "drums.mid").is_file()
