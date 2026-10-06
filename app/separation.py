"""Separate a mix into vocals, drums, bass, and other with Demucs htdemucs."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from app.errors import UserFacingError

logger = logging.getLogger(__name__)

STEMS = ("drums", "bass", "other", "vocals")

_separator = None
_separator_lock = threading.Lock()


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _is_memory_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, MemoryError) or "out of memory" in text or "not enough memory" in text


def get_separator():
    """Load htdemucs once and reuse it for later songs."""
    global _separator
    with _separator_lock:
        if _separator is None:
            try:
                from demucs.api import Separator
            except Exception as exc:
                raise UserFacingError(
                    "音源分离组件没有装好。请关掉程序，重新运行 start 脚本。"
                ) from exc
            try:
                _separator = Separator(
                    model="htdemucs",
                    device=pick_device(),
                    shifts=1,
                    overlap=0.25,
                    split=True,
                    segment=None,
                    jobs=0,
                    progress=False,
                )
            except Exception as exc:
                raise UserFacingError(
                    "音源分离模型下载或加载失败。第一次使用需要联网下载约 80MB 的模型。"
                    "请检查网络后，关掉程序再打开重试。"
                ) from exc
        return _separator


def _max_segment(separator) -> float | None:
    model = separator.model
    if hasattr(model, "max_allowed_segment"):
        value = float(model.max_allowed_segment)
        if value != float("inf"):
            return value
    segment = getattr(model, "segment", None)
    if segment is None:
        return None
    return float(segment)


def separate(path: Path, device: str | None = None, on_chunk=None) -> tuple[dict, int]:
    """Return stem tensors keyed by name, plus the model's sample rate.

    Tensors are shaped (channels, frames) on CPU.
    """
    separator = get_separator()
    chosen = device or pick_device()
    max_segment = _max_segment(separator)
    # Shorter chunks use less memory. Never ask the transformer for more than it was trained on.
    attempts: list[tuple[str, float | None]] = [(chosen, None)]
    if max_segment is not None:
        attempts.append((chosen, min(7.0, max_segment)))
        attempts.append(("cpu", min(5.0, max_segment)))
    else:
        attempts.append(("cpu", 7.0))

    last_error: Exception | None = None
    tried: set[tuple[str, float | None]] = set()
    for attempt_device, segment in attempts:
        key = (attempt_device, None if segment is None else round(segment, 2))
        if key in tried:
            continue
        tried.add(key)
        try:
            separator.update_parameter(
                device=attempt_device,
                segment=segment,
                callback=on_chunk,
                shifts=1,
                overlap=0.25,
                split=True,
            )
            _mix, stems = separator.separate_audio_file(Path(path))
            exported = {name: stem.detach().cpu() for name, stem in stems.items()}
            return exported, int(separator.samplerate)
        except UserFacingError:
            raise
        except Exception as exc:
            last_error = exc
            logger.warning("Separation failed on %s segment=%s: %s", attempt_device, segment, exc)
            if not _is_memory_error(exc) and attempt_device == "cpu":
                break
            continue

    if last_error is not None and _is_memory_error(last_error):
        raise UserFacingError("电脑内存不够，分离音轨失败了。请关掉其他软件，或换一首更短的歌再试。") from last_error
    raise UserFacingError(
        "分离音轨失败了。如果这是第一次使用，请确认电脑能上网（需要下载模型）。"
        "也可以把歌曲剪短一点再试。"
    ) from last_error
