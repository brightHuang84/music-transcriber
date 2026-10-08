"""Run separation, transcription, drums, tempo, chords, and MIDI export."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from app.audio_io import (
    convert_to_wav,
    duration_seconds,
    load_audio,
    rms,
    save_wav,
    waveform_peaks,
)
from app.chords import detect_chords
from app.drums import detect_drums
from app.errors import UserFacingError
from app.midi_export import write_combined_midi, write_stem_midi
from app.rhythm import analyze_rhythm
from app.separation import ensure_model, separate
from app.stems import DEFAULT_MODE, label_for, mode_spec, pitched_names
from app.transcription import transcribe

logger = logging.getLogger(__name__)

MAX_DURATION_SECONDS = 10 * 60
MIN_DURATION_SECONDS = 2.0


def _report(progress, percent: float, step: str, message: str) -> None:
    progress(int(max(0, min(100, percent))), step, message)


def _tensor_to_numpy(tensor) -> np.ndarray:
    array = tensor.detach().cpu().float().numpy()
    return np.asarray(array, dtype=np.float32)


def analyze(source: Path, work_dir: Path, progress, mode: str = DEFAULT_MODE) -> dict:
    """Analyze one song. `progress(percent, step, message)` is called along the way.

    `mode` is "fast" (vocals, drums, bass, other) or "fine" (those plus guitar and piano).
    """
    spec = mode_spec(mode)
    stem_names = tuple(spec["stems"])
    pitched = pitched_names(stem_names)
    work_dir.mkdir(parents=True, exist_ok=True)
    _report(progress, 3, "read", "正在读取音频…")
    mix_path = work_dir / "mix.wav"
    convert_to_wav(source, mix_path)
    mix, sample_rate = load_audio(mix_path)
    length = duration_seconds(mix, sample_rate)
    if length < MIN_DURATION_SECONDS:
        raise UserFacingError("音频太短了。请至少提供 2 秒，最好是一小段完整的旋律。")
    if length > MAX_DURATION_SECONDS:
        raise UserFacingError(
            "这首歌超过 10 分钟。在普通电脑上会跑很久，请先剪成较短的片段（3 分钟左右比较合适）。"
        )
    if rms(mix) < 1e-4:
        raise UserFacingError("这个文件几乎没有声音。请换一首歌试试。")

    names = "、".join(label_for(name) for name in stem_names)
    _report(progress, 6, "download", f"正在准备{spec['label']}模型…")

    def on_download(fraction: float, message: str) -> None:
        _report(progress, 6 + 10 * fraction, "download", message)

    # Download before the separation ticker, so the window shows bytes arriving
    # instead of a frozen "separating" line.
    ensure_model(spec["model"], on_progress=on_download)

    _report(
        progress,
        17,
        "separate",
        f"正在把歌曲分成 {names}。这一步最慢，请稍等…",
    )
    # The separator often reports progress only when a chunk finishes. A slow
    # ticker keeps the page from looking frozen on CPU.
    stop_ticker = threading.Event()

    def _tick() -> None:
        percent = 20.0
        while not stop_ticker.wait(1.5):
            percent = min(44.0, percent + 1.5)
            _report(
                progress,
                percent,
                "separate",
                f"正在分离音轨（{spec['label']}）…普通电脑会慢一些，页面可以开着，不用反复点击。",
            )

    ticker = threading.Thread(target=_tick, daemon=True)
    ticker.start()

    def on_chunk(info: dict) -> None:
        if info.get("state") != "end":
            return
        audio_length = float(info.get("audio_length") or 1)
        offset = float(info.get("segment_offset") or 0)
        fraction = min(1.0, max(0.0, offset / audio_length))
        _report(
            progress,
            18 + 28 * fraction,
            "separate",
            f"正在分离音轨（{spec['label']}）…普通电脑会慢一些，页面可以开着，不用反复点击。",
        )

    try:
        stem_tensors, model_rate = separate(mix_path, mode=spec["id"], on_chunk=on_chunk)
    finally:
        stop_ticker.set()
    if sample_rate != model_rate:
        import librosa

        mix = librosa.resample(mix, orig_sr=sample_rate, target_sr=model_rate, axis=-1).astype(np.float32)
        sample_rate = model_rate
        length = duration_seconds(mix, sample_rate)
    stem_audio: dict[str, np.ndarray] = {}
    stem_dir = work_dir / "stems"
    for name in stem_names:
        if name not in stem_tensors:
            raise UserFacingError("分离结果不完整，没有得到全部音轨。请重试一次。")
        audio = _tensor_to_numpy(stem_tensors[name])
        stem_audio[name] = audio
        save_wav(stem_dir / f"{name}.wav", audio, model_rate)

    _report(progress, 48, "rhythm", "正在听速度和节拍…")
    rhythm = analyze_rhythm(mix, stem_audio["drums"], model_rate)
    bpm = float(rhythm["bpm"])

    notes: dict[str, list] = {}
    for index, stem in enumerate(pitched):
        percent = 54 + int(28 * index / max(1, len(pitched)))
        _report(progress, percent, "notes", f"正在识别{label_for(stem)}的音高…")
        if rms(stem_audio[stem]) < 1e-3:
            notes[stem] = []
            continue
        notes[stem] = transcribe(stem_dir / f"{stem}.wav", stem, midi_tempo=bpm)

    _report(progress, 86, "drums", "正在分辨底鼓、军鼓、踩镲、叮叮镲、吊镲和通鼓…")
    hits = detect_drums(stem_audio["drums"], model_rate) if rms(stem_audio["drums"]) >= 1e-4 else []

    _report(progress, 92, "chords", "正在估计和弦…")
    harmony_audio = stem_audio[pitched[0]]
    for name in pitched[1:]:
        harmony_audio = harmony_audio + stem_audio[name]
    harmony = detect_chords(harmony_audio, model_rate, rhythm["beats"], length)

    _report(progress, 96, "midi", "正在生成 MIDI…")
    midi_dir = work_dir / "midi"
    for stem in pitched:
        write_stem_midi(midi_dir / f"{stem}.mid", stem, bpm, notes=notes[stem])
    write_stem_midi(midi_dir / "drums.mid", "drums", bpm, hits=hits)
    write_combined_midi(midi_dir / "all.mid", bpm, notes, hits)

    result = {
        "filename": source.name,
        "duration": round(length, 3),
        "sample_rate": model_rate,
        "mode": spec["id"],
        "mode_label": spec["label"],
        "bpm": rhythm["bpm"],
        "time_signature": rhythm["time_signature"],
        "beats": rhythm["beats"],
        "downbeats": rhythm["downbeats"],
        "chords": harmony["chords"],
        "key": harmony["key"],
        "mix_peaks": waveform_peaks(mix),
        "stems": {},
    }
    for stem in stem_names:
        entry = {
            "rms": round(rms(stem_audio[stem]), 5),
            "silent": rms(stem_audio[stem]) < 1e-3,
            "peaks": waveform_peaks(stem_audio[stem]),
        }
        if stem == "drums":
            entry["hits"] = hits
        else:
            entry["notes"] = notes[stem]
        result["stems"][stem] = entry
    _report(progress, 100, "done", "分析完成")
    logger.info(
        "Analyzed %s (%.1fs): %.1f BPM, notes=%s, hits=%d",
        source.name,
        length,
        bpm,
        {stem: len(notes[stem]) for stem in pitched},
        len(hits),
    )
    return result
