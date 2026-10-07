"""Load, convert, and save audio. ffmpeg covers mp3/m4a/flac on every OS."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from app.errors import UserFacingError

TARGET_SAMPLE_RATE = 44100
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac"}


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def ensure_ffmpeg() -> str:
    path = ffmpeg_path()
    if path is None:
        raise UserFacingError(
            "没有找到 ffmpeg，所以还不能读取歌曲。"
            "请先安装 ffmpeg，然后重新打开本程序。安装方法写在 README 里。"
        )
    return path


def convert_to_wav(source: Path, destination: Path, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
    """Convert any supported upload into a stereo WAV the rest of the pipeline can read."""
    ffmpeg = ensure_ffmpeg()
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "2",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise UserFacingError(
            "无法启动 ffmpeg，所以读不了这首歌。在 Ubuntu 上请重新运行 install.sh；在 Windows 或 macOS 上请先安装 ffmpeg，再打开程序。"
        ) from exc
    if completed.returncode != 0 or not destination.exists():
        raise UserFacingError(
            "这个文件读不出来。请确认它是完整的 mp3、wav、m4a 或 flac 歌曲，而不是视频或损坏的文件。"
        )


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Return float audio shaped (channels, frames) and the sample rate."""
    try:
        data, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    except Exception as exc:
        raise UserFacingError("音频文件打开失败。请换一个文件再试。") from exc
    audio = np.asarray(data, dtype=np.float32).T
    return audio, int(sample_rate)


def save_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write PCM-16 WAV. `audio` is (channels, frames) or (frames,)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if audio.ndim == 1:
        shaped = audio
    else:
        shaped = np.asarray(audio).T
    shaped = np.clip(np.asarray(shaped, dtype=np.float32), -1.0, 1.0)
    sf.write(str(path), shaped, int(sample_rate), subtype="PCM_16")


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    return audio.mean(axis=0).astype(np.float32, copy=False)


def rms(audio: np.ndarray) -> float:
    mono = to_mono(audio)
    if mono.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(mono))))


def duration_seconds(audio: np.ndarray, sample_rate: int) -> float:
    frames = audio.shape[-1]
    return float(frames) / float(sample_rate) if sample_rate else 0.0


def waveform_peaks(audio: np.ndarray, buckets: int = 480) -> list[float]:
    mono = np.abs(to_mono(audio))
    if mono.size == 0:
        return []
    count = int(min(buckets, mono.size))
    peaks: list[float] = []
    for chunk in np.array_split(mono, count):
        peaks.append(round(float(chunk.max()) if chunk.size else 0.0, 4))
    return peaks
