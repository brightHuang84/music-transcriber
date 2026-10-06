"""Checks that do not download the Demucs model."""

import time
import wave
from pathlib import Path

import numpy as np
import pretty_midi
from fastapi.testclient import TestClient

from app.audio_io import convert_to_wav, load_audio, save_wav
from app.chords import detect_chords
from app.drums import detect_drums
from app.midi_export import write_combined_midi, write_stem_midi
from app.notes import midi_to_name, midi_to_solfege
from app.rhythm import analyze_rhythm
from app.server import app
from app.synth import synthesize_example
from app.transcription import transcribe


def test_note_names():
    assert midi_to_name(60) == "C4"
    assert midi_to_name(64) == "E4"
    assert midi_to_name(36) == "C2"
    assert midi_to_solfege(64) == "mi"
    assert midi_to_solfege(61) == "#do"


def test_drum_hits_match_the_pattern():
    example = synthesize_example()
    hits = detect_drums(example["drum_audio"], example["sample_rate"])
    assert hits, "expected at least one drum hit"
    for hit in hits:
        assert hit["kind"] in {"kick", "snare", "hihat", "other"}
        assert 1 <= hit["velocity"] <= 127
        assert hit["time"] >= 0
    for expected in example["drums"]:
        matched = [
            hit
            for hit in hits
            if hit["kind"] == expected["kind"] and abs(hit["time"] - expected["time"]) <= 0.08
        ]
        assert matched, f"missing {expected} in {hits}"


def test_tempo_is_near_120_and_downbeats_land_on_kicks():
    example = synthesize_example()
    rhythm = analyze_rhythm(example["audio"], example["drum_audio"], example["sample_rate"])
    assert 100 <= rhythm["bpm"] <= 140
    assert len(rhythm["beats"]) >= 6
    assert rhythm["downbeats"], rhythm
    assert rhythm["downbeats"][0] <= 0.3
    if len(rhythm["downbeats"]) > 1:
        gap = rhythm["downbeats"][1] - rhythm["downbeats"][0]
        assert abs(gap - 2.0) < 0.2


def test_c_major_chord_and_key():
    example = synthesize_example()
    found = detect_chords(example["chord_audio"], example["sample_rate"], [0.0, 1.0], 2.0)
    symbols = [chord["symbol"] for chord in found["chords"]]
    assert "C" in symbols, found
    assert found["key"] is not None
    assert found["key"]["tonic"] == "C"
    assert found["key"]["mode"] == "major"


def test_melody_transcription_finds_the_written_pitches(tmp_path: Path):
    example = synthesize_example()
    path = tmp_path / "melody.wav"
    save_wav(path, example["melody_audio"], example["sample_rate"])
    notes = transcribe(path, "vocals", midi_tempo=120)
    for start, _end, pitch in example["melody"]:
        matched = [
            note
            for note in notes
            if note["pitch"] == pitch and abs(note["start"] - start) <= 0.12
        ]
        assert matched, f"missing pitch {pitch} at {start}: {notes}"
        assert matched[0]["name"]
        assert matched[0]["velocity"] >= 1
        assert matched[0]["duration"] > 0


def test_midi_files_round_trip(tmp_path: Path):
    notes = [
        {"pitch": 64, "start": 0.0, "end": 0.4, "velocity": 80, "name": "E4", "solfege": "mi", "duration": 0.4}
    ]
    bass = [
        {"pitch": 36, "start": 0.0, "end": 0.4, "velocity": 70, "name": "C2", "solfege": "do", "duration": 0.4}
    ]
    hits = [{"time": 0.5, "kind": "snare", "velocity": 100}]
    write_stem_midi(tmp_path / "vocals.mid", "vocals", 120, notes=notes)
    write_stem_midi(tmp_path / "drums.mid", "drums", 120, hits=hits)
    write_combined_midi(tmp_path / "all.mid", 120, {"vocals": notes, "bass": bass, "other": []}, hits)
    vocals = pretty_midi.PrettyMIDI(str(tmp_path / "vocals.mid"))
    assert vocals.instruments[0].notes[0].pitch == 64
    drums = pretty_midi.PrettyMIDI(str(tmp_path / "drums.mid"))
    assert drums.instruments[0].is_drum
    assert drums.instruments[0].notes[0].pitch == 38
    combined = pretty_midi.PrettyMIDI(str(tmp_path / "all.mid"))
    names = {instrument.name for instrument in combined.instruments}
    assert {"Vocals", "Bass", "Drums"} <= names


def test_ffmpeg_reads_mp3(tmp_path: Path):
    wav_path = tmp_path / "tone.wav"
    sample_rate = 44100
    frames = sample_rate
    tone = (0.2 * np.sin(2 * np.pi * 440 * np.arange(frames) / sample_rate)).astype(np.float32)
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((tone * 32767).astype(np.int16).tobytes())
    mp3_path = tmp_path / "tone.mp3"
    import subprocess

    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(wav_path), str(mp3_path)],
        check=True,
    )
    converted = tmp_path / "back.wav"
    convert_to_wav(mp3_path, converted)
    audio, rate = load_audio(converted)
    assert rate == 44100
    assert audio.shape[0] == 2
    assert audio.shape[1] > sample_rate // 2


def test_api_rejects_unknown_extension(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MUSIC_ANALYZER_DATA", str(tmp_path))
    client = TestClient(app)
    response = client.post("/api/analyze", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400
    assert "mp3" in response.json()["detail"]


def test_api_short_audio_has_a_friendly_error(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MUSIC_ANALYZER_DATA", str(tmp_path))
    wav_path = tmp_path / "short.wav"
    sample_rate = 44100
    frames = int(0.4 * sample_rate)
    tone = (0.2 * np.sin(2 * np.pi * 440 * np.arange(frames) / sample_rate)).astype(np.float32)
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((tone * 32767).astype(np.int16).tobytes())
    client = TestClient(app)
    with wav_path.open("rb") as handle:
        response = client.post("/api/analyze", files={"file": ("short.wav", handle, "audio/wav")})
    assert response.status_code == 200
    job_id = response.json()["id"]
    status = {}
    for _ in range(80):
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["status"] in {"done", "error"}:
            break
        time.sleep(0.1)
    assert status["status"] == "error"
    assert "太短" in status["message"]


def test_home_page_is_chinese():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "听音识谱" in response.text
