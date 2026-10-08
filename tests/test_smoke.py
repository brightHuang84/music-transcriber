"""End-to-end check on a short synthesized clip. The first run downloads Demucs."""

from pathlib import Path

import json

import pretty_midi
import pytest

from app.pipeline import analyze
from app.synth import synthesize_example

pytestmark = pytest.mark.slow


def test_pipeline_on_synthesized_clip(tmp_path: Path):
    source = tmp_path / "example.wav"
    example = synthesize_example(source)
    updates: list[tuple[int, str, str]] = []

    def progress(percent: int, step: str, message: str) -> None:
        updates.append((percent, step, message))

    result = analyze(source, tmp_path / "work", progress, mode="fast")
    json.dumps(result)
    assert updates[-1][0] == 100
    assert result["duration"] == pytest.approx(example["duration"], abs=0.1)
    assert result["bpm"] > 0
    assert result["beats"]
    assert result["mode"] == "fast"
    assert set(result["stems"]) == {"vocals", "drums", "bass", "other"}
    for name in ("vocals", "drums", "bass", "other"):
        audio = tmp_path / "work" / "stems" / f"{name}.wav"
        assert audio.exists() and audio.stat().st_size > 1000
        stem = result["stems"][name]
        assert "peaks" in stem
        if name == "drums":
            for hit in stem["hits"]:
                assert hit["kind"] in {
                    "kick",
                    "snare",
                    "hat_closed",
                    "hat_open",
                    "ride",
                    "crash",
                    "tom_high",
                    "tom_mid",
                    "tom_floor",
                }
                assert "time" in hit and "velocity" in hit
        else:
            for note in stem["notes"]:
                assert {"pitch", "start", "end", "duration", "velocity", "name"} <= set(note)
    for name in ("vocals", "bass", "other", "drums", "all"):
        midi = pretty_midi.PrettyMIDI(str(tmp_path / "work" / "midi" / f"{name}.mid"))
        if name == "all":
            assert midi.instruments
            continue
        if name == "drums":
            expected = result["stems"]["drums"]["hits"]
        else:
            expected = result["stems"][name]["notes"]
        # A stem that is only a noise floor has no notes. pretty_midi drops an
        # empty track on read, which is the file we want in that case.
        if expected:
            assert midi.instruments
            assert midi.instruments[0].notes
        else:
            assert not midi.instruments or not midi.instruments[0].notes
