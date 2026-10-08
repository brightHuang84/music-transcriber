"""Measure stem SDR and drum-hit scores. Audio stays under /tmp/sep_eval.

Ground truth is a short General MIDI arrangement rendered with FluidSynth
(piano, guitar, violin, string pad, bass, choir, and a drum kit, including
hits that land together). MUSDB18 is not downloaded: it is not freely
redistributable.

Run from the repo root:

    .venv/bin/python tools/eval_separation.py render
    .venv/bin/python tools/eval_separation.py demucs htdemucs
    .venv/bin/python tools/eval_separation.py demucs htdemucs_6s
    .venv/bin/python tools/eval_separation.py report
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path("/tmp/sep_eval")
SR = 44100
SOUNDFONTS = (
    Path("/usr/share/sounds/sf2/FluidR3_GM.sf2"),
    Path("/usr/share/sounds/sf2/default-GM.sf2"),
    Path("/usr/share/soundfonts/FluidR3_GM.sf2"),
)

# Fine-class names used by the app, plus the six DrumSep families.
FAMILY = {
    "kick": "kick",
    "snare": "snare",
    "hat_closed": "hat",
    "hat_open": "hat",
    "tom_high": "tom",
    "tom_mid": "tom",
    "tom_floor": "tom",
    "ride": "ride",
    "crash": "crash",
    "hat": "hat",
    "tom": "tom",
    "hh": "hat",
    "toms": "tom",
}


def soundfont() -> Path:
    if shutil.which("fluidsynth") is None:
        raise SystemExit("fluidsynth is not installed")
    for path in SOUNDFONTS:
        if path.is_file():
            return path
    raise SystemExit("FluidR3_GM.sf2 was not found")


def _add(notes: list, start: float, end: float, pitch: int, velocity: int = 92) -> None:
    notes.append((round(start, 4), round(end, 4), int(pitch), int(velocity)))


def arrangement() -> dict:
    """Sixteen seconds. Instruments overlap on purpose, the way a film cue does."""
    piano, guitar, violin, strings, bass, vocals = [], [], [], [], [], []
    drums: list[tuple] = []

    piano_chords = [
        (0.0, (60, 64, 67)),
        (2.0, (53, 57, 60)),
        (4.0, (55, 59, 62)),
        (6.0, (60, 64, 69)),
        (8.0, (57, 60, 64)),
        (10.0, (62, 65, 69)),
        (12.0, (55, 59, 62)),
        (14.0, (60, 64, 67, 72)),
    ]
    for start, chord in piano_chords:
        for pitch in chord:
            _add(piano, start, start + 1.7, pitch, 84)

    guitar_pattern = (64, 67, 71, 76)
    for bar in range(8):
        for step, pitch in enumerate(guitar_pattern):
            start = bar * 2.0 + 0.25 + step * 0.42
            _add(guitar, start, start + 0.38, pitch, 78)

    violin_line = (76, 79, 81, 84, 81, 79, 77, 74)
    for index, pitch in enumerate(violin_line):
        _add(violin, 0.35 + index * 1.9, 0.35 + index * 1.9 + 1.55, pitch, 80)

    for start, root in ((0.0, 48), (4.0, 41), (8.0, 43), (12.0, 48)):
        _add(strings, start, start + 3.8, root, 62)
        _add(strings, start, start + 3.8, root + 7, 58)

    bass_line = (36, 36, 41, 41, 43, 43, 36, 38)
    for index, pitch in enumerate(bass_line):
        _add(bass, index * 2.0, index * 2.0 + 1.6, pitch, 96)

    vocal_line = (72, 74, 76, 79, 77, 74, 72, 71)
    for index, pitch in enumerate(vocal_line):
        _add(vocals, 0.15 + index * 1.9, 0.15 + index * 1.9 + 1.45, pitch, 86)

    # (time, midi note, kind). Several share a time on purpose.
    kit = [
        (0.00, 36, "kick"),
        (0.00, 42, "hat_closed"),
        (0.50, 42, "hat_closed"),
        (1.00, 38, "snare"),
        (1.00, 42, "hat_closed"),
        (1.50, 42, "hat_closed"),
        (2.00, 36, "kick"),
        (2.00, 42, "hat_closed"),
        (2.50, 46, "hat_open"),
        (3.00, 38, "snare"),
        (3.00, 49, "crash"),
        (3.50, 36, "kick"),
        (3.50, 51, "ride"),
        (4.00, 36, "kick"),
        (4.00, 42, "hat_closed"),
        (4.00, 50, "tom_high"),
        (4.50, 42, "hat_closed"),
        (5.00, 38, "snare"),
        (5.00, 42, "hat_closed"),
        (5.50, 51, "ride"),
        (6.00, 36, "kick"),
        (6.00, 38, "snare"),
        (6.50, 42, "hat_closed"),
        (7.00, 47, "tom_mid"),
        (7.50, 36, "kick"),
        (7.50, 43, "tom_floor"),
        (8.00, 36, "kick"),
        (8.00, 42, "hat_closed"),
        (8.50, 49, "crash"),
        (9.00, 38, "snare"),
        (9.00, 51, "ride"),
        (9.50, 46, "hat_open"),
        (10.00, 36, "kick"),
        (10.00, 42, "hat_closed"),
        (10.50, 50, "tom_high"),
        (11.00, 38, "snare"),
        (11.00, 42, "hat_closed"),
        (11.50, 36, "kick"),
        (12.00, 36, "kick"),
        (12.00, 49, "crash"),
        (12.50, 42, "hat_closed"),
        (13.00, 38, "snare"),
        (13.00, 43, "tom_floor"),
        (13.50, 51, "ride"),
        (14.00, 36, "kick"),
        (14.00, 42, "hat_closed"),
        (14.50, 42, "hat_closed"),
        (15.00, 38, "snare"),
        (15.00, 46, "hat_open"),
        (15.50, 36, "kick"),
        (15.50, 51, "ride"),
    ]
    for when, pitch, kind in kit:
        drums.append((when, pitch, kind))

    return {
        "vocals": {"program": 53, "drum": False, "notes": vocals},
        "bass": {"program": 33, "drum": False, "notes": bass},
        "guitar": {"program": 25, "drum": False, "notes": guitar},
        "piano": {"program": 0, "drum": False, "notes": piano},
        "violin": {"program": 40, "drum": False, "notes": violin},
        "strings": {"program": 48, "drum": False, "notes": strings},
        "drums": {"program": 0, "drum": True, "notes": drums},
    }


def _render_midi(path: Path, program: int, is_drum: bool, notes: list) -> None:
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    instrument = pretty_midi.Instrument(program=program, is_drum=is_drum, name="stem")
    for item in notes:
        if is_drum:
            start, pitch, _kind = item
            end = start + 0.45
            velocity = 104
        else:
            start, end, pitch, velocity = item
        instrument.notes.append(
            pretty_midi.Note(velocity=int(velocity), pitch=int(pitch), start=float(start), end=float(end))
        )
    midi.instruments.append(instrument)
    score = path.with_suffix(".mid")
    midi.write(str(score))
    subprocess.run(
        [
            "fluidsynth",
            "-ni",
            "-g",
            "0.6",
            str(soundfont()),
            str(score),
            "-F",
            str(path),
            "-r",
            str(SR),
        ],
        check=True,
        capture_output=True,
    )


def _load(path: Path) -> np.ndarray:
    import soundfile as sf

    data, rate = sf.read(str(path), always_2d=True, dtype="float32")
    if int(rate) != SR:
        raise SystemExit(f"{path} is {rate} Hz, expected {SR}")
    return np.ascontiguousarray(data.T)


def _save(path: Path, audio: np.ndarray) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    shaped = audio.T if audio.ndim == 2 else audio
    sf.write(str(path), shaped, SR, subtype="PCM_16")


def render() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    raw_dir = ROOT / "raw"
    raw_dir.mkdir(exist_ok=True)
    parts = arrangement()
    audio: dict[str, np.ndarray] = {}
    for name, spec in parts.items():
        path = raw_dir / f"{name}.wav"
        print(f"rendering {name}", flush=True)
        _render_midi(path, spec["program"], spec["drum"], spec["notes"])
        audio[name] = _load(path)
    length = min(item.shape[-1] for item in audio.values())
    # Keep a little tail so the last hit is inside the file, then stop at 16.2 s.
    length = min(length, int(16.2 * SR))
    audio = {name: item[:, :length] for name, item in audio.items()}
    mix = np.zeros_like(next(iter(audio.values())))
    for item in audio.values():
        mix = mix + item
    peak = float(np.max(np.abs(mix))) + 1e-8
    scale = 0.89 / peak
    audio = {name: (item * scale).astype(np.float32) for name, item in audio.items()}
    mix = (mix * scale).astype(np.float32)
    ref = ROOT / "ref"
    for name, item in audio.items():
        _save(ref / f"{name}.wav", item)
    other6 = audio["violin"] + audio["strings"]
    other4 = other6 + audio["piano"] + audio["guitar"]
    _save(ref / "other6.wav", other6.astype(np.float32))
    _save(ref / "other4.wav", other4.astype(np.float32))
    _save(ROOT / "mix.wav", mix)
    hits = [{"time": when, "kind": kind} for when, _pitch, kind in parts["drums"]["notes"] if when < length / SR]
    (ROOT / "drum_hits.json").write_text(json.dumps(hits, indent=2), encoding="utf-8")
    print(f"wrote {ROOT / 'mix.wav'} ({length / SR:.2f}s)", flush=True)


def mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float64)
    return np.mean(audio, axis=0).astype(np.float64)


def _shift(audio: np.ndarray, lag: int) -> np.ndarray:
    if lag == 0:
        return audio
    out = np.zeros_like(audio)
    if lag > 0:
        out[..., lag:] = audio[..., : audio.shape[-1] - lag]
    else:
        out[..., : audio.shape[-1] + lag] = audio[..., -lag:]
    return out


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    est = mono(estimate)
    ref = mono(reference)
    n = min(est.size, ref.size)
    est = est[:n]
    ref = ref[:n]
    est = est - est.mean()
    ref = ref - ref.mean()
    ref_energy = float(np.dot(ref, ref))
    if ref_energy < 1e-12:
        return float("nan")
    scale = float(np.dot(ref, est)) / ref_energy
    target = scale * ref
    noise = est - target
    return float(10.0 * np.log10((np.dot(target, target) + 1e-12) / (np.dot(noise, noise) + 1e-12)))


def best_lag(estimate: np.ndarray, reference: np.ndarray, max_lag: int = 2205) -> int:
    from scipy.signal import correlate

    est = mono(estimate)
    ref = mono(reference)
    n = min(est.size, ref.size, SR * 8)
    est = est[:n] - np.mean(est[:n])
    ref = ref[:n] - np.mean(ref[:n])
    corr = correlate(est, ref, mode="full", method="fft")
    mid = n - 1
    lo = max(0, mid - max_lag)
    hi = min(corr.size, mid + max_lag + 1)
    return int(lo + int(np.argmax(corr[lo:hi])) - mid)


def bss_sdr(estimates: dict[str, np.ndarray], references: dict[str, np.ndarray], lag: int) -> dict[str, float]:
    from mir_eval.separation import bss_eval_sources

    names = [name for name in references if name in estimates]
    ref = np.stack([mono(_shift(references[name], 0)) for name in names])
    est = np.stack([mono(_shift(estimates[name], lag)) for name in names])
    n = min(ref.shape[1], est.shape[1])
    ref = ref[:, :n]
    est = est[:, :n]
    sdr, _sir, _sar, _perm = bss_eval_sources(ref, est, compute_permutation=False)
    return {name: float(value) for name, value in zip(names, sdr)}


def references_for(stems: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    six = "piano" in stems and "guitar" in stems
    ref = {name: _load(ROOT / "ref" / f"{name}.wav") for name in ("vocals", "drums", "bass", "guitar", "piano")}
    ref["other"] = _load(ROOT / "ref" / ("other6.wav" if six else "other4.wav"))
    ref["violin"] = _load(ROOT / "ref" / "violin.wav")
    ref["strings"] = _load(ROOT / "ref" / "strings.wav")
    return ref


def score_separation(name: str, stems: dict[str, np.ndarray], seconds: float, download_mb: float | None) -> dict:
    refs = references_for(stems)
    paired = {key: stems[key] for key in ("vocals", "drums", "bass", "other") if key in stems}
    if "piano" in stems:
        paired["piano"] = stems["piano"]
    if "guitar" in stems:
        paired["guitar"] = stems["guitar"]
    ref_sum = np.zeros_like(next(iter(paired.values())))
    est_sum = np.zeros_like(ref_sum)
    for key, est in paired.items():
        ref = refs[key]
        n = min(ref.shape[-1], est.shape[-1], ref_sum.shape[-1])
        ref_sum[..., :n] += ref[..., :n]
        est_sum[..., :n] += est[..., :n]
    lag = best_lag(est_sum, ref_sum)
    si = {}
    for key, est in paired.items():
        si[key] = round(si_sdr(_shift(est, lag), refs[key]), 2)
    sdr = {key: round(value, 2) for key, value in bss_sdr(paired, {k: refs[k] for k in paired}, lag).items()}
    bleed = {}
    for est_name, est in paired.items():
        row = {}
        for ref_name in ("vocals", "drums", "bass", "guitar", "piano", "violin", "strings"):
            row[ref_name] = round(si_sdr(_shift(est, lag), refs[ref_name]), 2)
        bleed[est_name] = row
    violin_home = max(bleed, key=lambda stem: bleed[stem]["violin"])
    record = {
        "name": name,
        "seconds": round(seconds, 2),
        "audio_seconds": round(next(iter(paired.values())).shape[-1] / SR, 2),
        "download_mb": download_mb,
        "lag_samples": int(lag),
        "si_sdr": si,
        "sdr": sdr,
        "bleed_si_sdr": bleed,
        "violin_highest_in": violin_home,
        "violin_si_sdr_in_other": bleed.get("other", {}).get("violin"),
        "violin_si_sdr_in_piano": bleed.get("piano", {}).get("violin"),
        "violin_si_sdr_in_guitar": bleed.get("guitar", {}).get("violin"),
    }
    _store(name, record)
    print(json.dumps({"name": name, "si_sdr": si, "sdr": sdr, "seconds": record["seconds"], "violin_in": violin_home}, indent=2))
    return record


def _store(name: str, record: dict) -> None:
    path = ROOT / "results.json"
    blob = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    blob[name] = record
    path.write_text(json.dumps(blob, indent=2), encoding="utf-8")


def _as_numpy(stem) -> np.ndarray:
    if hasattr(stem, "detach"):
        stem = stem.detach().cpu().float().numpy()
    audio = np.asarray(stem, dtype=np.float32)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    if audio.shape[0] != 2 and audio.shape[-1] == 2:
        audio = audio.T
    return audio


def run_demucs(model: str, shifts: int, overlap: float) -> None:
    import torch
    from demucs.api import Separator

    torch.set_num_threads(4)
    mix = ROOT / "mix.wav"
    if not mix.exists():
        render()
    label = model if shifts == 1 and abs(overlap - 0.25) < 1e-6 else f"{model}_s{shifts}_o{overlap}"
    print(f"separating with {label}", flush=True)
    started = time.perf_counter()
    separator = Separator(
        model=model,
        device="cpu",
        shifts=shifts,
        overlap=overlap,
        split=True,
        segment=None,
        jobs=0,
        progress=True,
    )
    _mix, stems = separator.separate_audio_file(mix)
    elapsed = time.perf_counter() - started
    exported = {name: _as_numpy(stem) for name, stem in stems.items()}
    out = ROOT / "est" / label
    for name, audio in exported.items():
        _save(out / f"{name}.wav", audio)
    sizes = {
        "htdemucs": 84,
        "htdemucs_6s": 55,
        "htdemucs_ft": 320,
    }
    score_separation(label, exported, elapsed, sizes.get(model))
    write_spectrograms(label, exported)


def run_roformer() -> None:
    import torch
    import yaml
    from bs_roformer.download import ensure_model_assets
    from bs_roformer.inference import SafeLoaderWithTuple
    from bs_roformer.utils import demix_track, get_model_from_config
    from ml_collections import ConfigDict

    torch.set_num_threads(4)
    mix_path = ROOT / "mix.wav"
    if not mix_path.exists():
        render()
    print("loading BS-RoFormer-SW", flush=True)
    ckpt, config_path = ensure_model_assets()
    download_mb = round(Path(ckpt).stat().st_size / 1e6, 1)
    with open(config_path, encoding="utf-8") as handle:
        config = ConfigDict(yaml.load(handle, Loader=SafeLoaderWithTuple))
    model = get_model_from_config("bs_roformer", config)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    model.to("cpu")
    mix = torch.tensor(_load(mix_path), dtype=torch.float32)
    print("running BS-RoFormer-SW", flush=True)
    started = time.perf_counter()
    stems, _chunk = demix_track(config, model, mix, "cpu")
    elapsed = time.perf_counter() - started
    exported = {name: _as_numpy(audio) for name, audio in stems.items()}
    out = ROOT / "est" / "bs_roformer_sw"
    for name, audio in exported.items():
        _save(out / f"{name}.wav", audio)
    score_separation("bs_roformer_sw", exported, elapsed, download_mb)
    write_spectrograms("bs_roformer_sw", exported)


def score_hits(expected: list[dict], detected: list[dict], tolerance: float, family: bool) -> dict:
    def key(hit: dict) -> str:
        kind = hit["kind"]
        if family:
            return FAMILY.get(kind, kind)
        return kind

    classes = sorted({key(hit) for hit in expected} | {key(hit) for hit in detected})
    per: dict[str, dict] = {}
    for kind in classes:
        exp = [hit for hit in expected if key(hit) == kind]
        det = [hit for hit in detected if key(hit) == kind]
        used: set[int] = set()
        matched = 0
        for hit in exp:
            best = None
            best_dt = tolerance
            for index, other in enumerate(det):
                if index in used:
                    continue
                delta = abs(float(other["time"]) - float(hit["time"]))
                if delta <= best_dt:
                    best_dt = delta
                    best = index
            if best is not None:
                used.add(best)
                matched += 1
        precision = matched / len(det) if det else float("nan")
        recall = matched / len(exp) if exp else float("nan")
        per[kind] = {
            "expected": len(exp),
            "detected": len(det),
            "matched": matched,
            "precision": None if precision != precision else round(precision, 3),
            "recall": None if recall != recall else round(recall, 3),
        }
    exp_n = sum(item["expected"] for item in per.values())
    det_n = sum(item["detected"] for item in per.values())
    matched = sum(item["matched"] for item in per.values())
    return {
        "micro_precision": round(matched / det_n, 3) if det_n else None,
        "micro_recall": round(matched / exp_n, 3) if exp_n else None,
        "per_class": per,
    }


def _onsets(audio: np.ndarray, sample_rate: int) -> list[float]:
    import librosa

    wave = mono(audio).astype(np.float32)
    if float(np.sqrt(np.mean(wave**2))) < 1e-4:
        return []
    times = librosa.onset.onset_detect(
        y=wave,
        sr=sample_rate,
        units="time",
        backtrack=False,
        delta=0.12,
        wait=4,
    )
    return [round(float(item), 4) for item in times]


def run_drums() -> None:
    from app.drums import detect_drums, split_family

    drum_path = ROOT / "ref" / "drums.wav"
    if not drum_path.exists():
        render()
    expected = json.loads((ROOT / "drum_hits.json").read_text(encoding="utf-8"))
    audio = _load(drum_path)
    print("ADTOF on clean GM drums", flush=True)
    started = time.perf_counter()
    adtof = detect_drums(audio, SR)
    adtof_seconds = time.perf_counter() - started
    report = {
        "adtof_clean": {
            "seconds": round(adtof_seconds, 2),
            "family": score_hits(expected, adtof, 0.05, family=True),
            "fine": score_hits(expected, adtof, 0.05, family=False),
        }
    }
    separated = ROOT / "est"
    if separated.exists():
        for folder in sorted(separated.iterdir()):
            drum = folder / "drums.wav"
            if not drum.exists():
                continue
            hits = detect_drums(_load(drum), SR)
            report[f"adtof_on_{folder.name}"] = {
                "family": score_hits(expected, hits, 0.05, family=True),
                "fine": score_hits(expected, hits, 0.05, family=False),
            }

    print("DrumSep", flush=True)
    import torch
    from mdxnet_infer import MDX23CInference

    torch.set_num_threads(4)
    started = time.perf_counter()
    engine = MDX23CInference.from_pretrained("drumsep-6stem", device="cpu")
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    pieces = engine.separate(audio.T, sample_rate=SR, progress=True)
    sep_seconds = time.perf_counter() - started
    ckpt = None
    for path in Path.home().glob(".cache/**/*DrumSep*.ckpt"):
        ckpt = path
        break
    download_mb = round(ckpt.stat().st_size / 1e6, 1) if ckpt else None
    drumsep_hits: list[dict] = []
    stem_dir = ROOT / "est" / "drumsep"
    mapping = {"kick": "kick", "snare": "snare", "toms": "tom", "hh": "hat", "ride": "ride", "crash": "crash"}
    for stem_name, kind in mapping.items():
        piece = pieces[stem_name]
        _save(stem_dir / f"{stem_name}.wav", _as_numpy(piece))
        for when in _onsets(_as_numpy(piece), SR):
            if kind == "tom":
                start = int(when * SR)
                segment = mono(_as_numpy(piece))[start : start + int(0.5 * SR)].astype(np.float32)
                label = split_family("tom", segment, SR)
            elif kind == "hat":
                start = int(when * SR)
                segment = mono(_as_numpy(piece))[start : start + int(0.5 * SR)].astype(np.float32)
                label = split_family("hat", segment, SR)
            else:
                label = kind
            drumsep_hits.append({"time": when, "kind": label, "velocity": 90})
    fused = _fuse(adtof, drumsep_hits)
    report["drumsep"] = {
        "load_seconds": round(load_seconds, 2),
        "seconds": round(sep_seconds, 2),
        "download_mb": download_mb,
        "onsets": {
            "family": score_hits(expected, drumsep_hits, 0.05, family=True),
            "fine": score_hits(expected, drumsep_hits, 0.05, family=False),
        },
        "fused_with_adtof": {
            "family": score_hits(expected, fused, 0.05, family=True),
            "fine": score_hits(expected, fused, 0.05, family=False),
        },
    }
    path = ROOT / "drums.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _fuse(adtof: list[dict], drumsep: list[dict]) -> list[dict]:
    """Keep a hit when either detector hears it, without duplicating the same class."""
    merged = [dict(hit) for hit in adtof]
    for hit in drumsep:
        if any(abs(hit["time"] - other["time"]) <= 0.04 and FAMILY.get(hit["kind"], hit["kind"]) == FAMILY.get(other["kind"], other["kind"]) for other in merged):
            continue
        merged.append(dict(hit))
    merged.sort(key=lambda item: (item["time"], item["kind"]))
    return merged


def _png(path: Path, rgb: np.ndarray) -> None:
    height, width, _channels = rgb.shape

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    blob = b"\x89PNG\r\n\x1a\n"
    blob += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    blob += chunk(b"IDAT", zlib.compress(raw, 9))
    blob += chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)


def spectrogram_image(audio: np.ndarray, width: int = 480, height: int = 180) -> np.ndarray:
    wave = mono(audio).astype(np.float32)
    n_fft = 2048
    hop = max(1, wave.size // width)
    frames = []
    window = np.hanning(n_fft).astype(np.float32)
    for index in range(width):
        start = index * hop
        clip = wave[start : start + n_fft]
        if clip.size < n_fft:
            clip = np.pad(clip, (0, n_fft - clip.size))
        spectrum = np.abs(np.fft.rfft(clip * window))
        frames.append(spectrum)
    grid = np.stack(frames, axis=1)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / SR)
    lo = int(np.searchsorted(freqs, 40))
    hi = int(np.searchsorted(freqs, 12000))
    grid = grid[lo:hi]
    # Log-frequency rows.
    positions = np.geomspace(1, grid.shape[0], height).astype(np.int32) - 1
    grid = grid[np.clip(positions, 0, grid.shape[0] - 1)]
    grid = np.log10(grid + 1e-6)
    lo_v = float(np.percentile(grid, 5))
    hi_v = float(np.percentile(grid, 99))
    gray = np.clip((grid - lo_v) / (hi_v - lo_v + 1e-8), 0, 1)
    # Low energy is ink blue, high energy is warm paper-white. Time runs left to right,
    # low frequencies are at the bottom.
    gray = gray[::-1]
    color = np.zeros((height, width, 3), dtype=np.uint8)
    color[..., 0] = (30 + 210 * gray).astype(np.uint8)
    color[..., 1] = (36 + 170 * np.sqrt(gray)).astype(np.uint8)
    color[..., 2] = (64 + 80 * (1 - gray)).astype(np.uint8)
    return color


def write_spectrograms(label: str, stems: dict[str, np.ndarray]) -> None:
    fig = ROOT / "figs" / label
    for name, audio in stems.items():
        _png(fig / f"{name}.png", spectrogram_image(audio))
    refs = references_for(stems)
    for name in stems:
        if name not in refs:
            continue
        ref_img = spectrogram_image(refs[name])
        est_img = spectrogram_image(stems[name])
        gap = np.full((8, ref_img.shape[1], 3), 240, dtype=np.uint8)
        _png(fig / f"{name}_vs_ref.png", np.concatenate([ref_img, gap, est_img], axis=0))


def owner_excerpt(seconds: float = 20.0, start: float = 48.0) -> None:
    """Spectral-flux correlation between stems of the owner's song. No ground truth."""
    import librosa

    source = Path("/tmp/user_song_extract/user_song.mp3")
    if not source.exists():
        raise SystemExit(f"missing {source}")
    dest = ROOT / "owner"
    dest.mkdir(parents=True, exist_ok=True)
    excerpt = dest / "excerpt.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(start),
            "-t",
            str(seconds),
            "-i",
            str(source),
            "-ac",
            "2",
            "-ar",
            str(SR),
            str(excerpt),
        ],
        check=True,
    )
    print(f"wrote {excerpt}", flush=True)


def owner_bleed(folder: Path, label: str) -> dict:
    import librosa

    names = [path.stem for path in sorted(folder.glob("*.wav"))]
    envs = {}
    for name in names:
        wave = mono(_load(folder / f"{name}.wav")).astype(np.float32)
        envs[name] = librosa.onset.onset_strength(y=wave, sr=SR)
    length = min(item.size for item in envs.values())
    pairs = {}
    keys = list(envs)
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            a = envs[left][:length]
            b = envs[right][:length]
            a = (a - a.mean()) / (a.std() + 1e-8)
            b = (b - b.mean()) / (b.std() + 1e-8)
            pairs[f"{left}|{right}"] = round(float(np.mean(a * b)), 3)
    record = {"label": label, "onset_correlation": pairs}
    path = ROOT / "owner_bleed.json"
    blob = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    blob[label] = record
    path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return record


def ensemble(left_name: str, right_name: str, weights: tuple[float, float] = (0.5, 0.5)) -> None:
    """Average two saved separations. No extra model time."""
    left = ROOT / "est" / left_name
    right = ROOT / "est" / right_name
    names = sorted(path.stem for path in left.glob("*.wav") if (right / path.name).exists())
    if not names:
        raise SystemExit(f"no shared stems in {left} and {right}")
    mixed = {}
    for name in names:
        a = _load(left / f"{name}.wav")
        b = _load(right / f"{name}.wav")
        n = min(a.shape[-1], b.shape[-1])
        mixed[name] = (weights[0] * a[..., :n] + weights[1] * b[..., :n]).astype(np.float32)
    label = f"avg_{left_name}_{right_name}"
    out = ROOT / "est" / label
    for name, audio in mixed.items():
        _save(out / f"{name}.wav", audio)
    score_separation(label, mixed, seconds=0.0, download_mb=None)


def report() -> None:
    results = json.loads((ROOT / "results.json").read_text(encoding="utf-8")) if (ROOT / "results.json").exists() else {}
    drums = json.loads((ROOT / "drums.json").read_text(encoding="utf-8")) if (ROOT / "drums.json").exists() else {}
    print("## separation")
    for name, record in results.items():
        print(name, "time", record.get("seconds"), "MB", record.get("download_mb"))
        print("  SI-SDR", record.get("si_sdr"))
        print("  SDR   ", record.get("sdr"))
        print("  violin highest in", record.get("violin_highest_in"), "other", record.get("violin_si_sdr_in_other"))
    print("## drums")
    print(json.dumps(drums, indent=2)[:4000])


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("render")
    dem = sub.add_parser("demucs")
    dem.add_argument("model")
    dem.add_argument("--shifts", type=int, default=1)
    dem.add_argument("--overlap", type=float, default=0.25)
    sub.add_parser("roformer")
    sub.add_parser("drums")
    sub.add_parser("owner-excerpt")
    bleed = sub.add_parser("owner-bleed")
    bleed.add_argument("folder")
    bleed.add_argument("label")
    ens = sub.add_parser("ensemble")
    ens.add_argument("left")
    ens.add_argument("right")
    sub.add_parser("report")
    args = parser.parse_args()
    if args.cmd == "render":
        render()
    elif args.cmd == "demucs":
        run_demucs(args.model, args.shifts, args.overlap)
    elif args.cmd == "roformer":
        run_roformer()
    elif args.cmd == "drums":
        run_drums()
    elif args.cmd == "owner-excerpt":
        owner_excerpt()
    elif args.cmd == "owner-bleed":
        owner_bleed(Path(args.folder), args.label)
    elif args.cmd == "ensemble":
        ensemble(args.left, args.right)
    elif args.cmd == "report":
        report()


if __name__ == "__main__":
    main()
