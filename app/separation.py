"""Separate a mix with Demucs. 快速 is four stems; 精细 adds piano and guitar."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from app.errors import UserFacingError
from app.stems import MODES, mode_spec

logger = logging.getLogger(__name__)

_separators: dict[str, object] = {}
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


def _download_bar(on_progress, label: str, size_mb: int):
    """A tqdm subclass Hugging Face will drive while a weight file downloads."""
    from tqdm import tqdm

    class DownloadBar(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["file"] = open("/dev/null", "w")  # noqa: SIM115 — closed in close()
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            shown = super().update(n)
            total = float(self.total or 0)
            if total > 0 and on_progress is not None:
                done_mb = self.n / 1e6
                total_mb = total / 1e6
                on_progress(
                    min(1.0, self.n / total),
                    f"正在下载{label}模型… {done_mb:.0f}/{total_mb:.0f} MB。"
                    f"大约 {size_mb}MB，下完留在这台电脑上，下次不用再下。",
                )
            return shown

        def close(self):
            sink = getattr(self, "fp", None)
            super().close()
            if sink is not None and hasattr(sink, "close"):
                sink.close()

    return DownloadBar


def _weights_cached(repo_id: str, filename: str) -> bool:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        hf_hub_download(repo_id, filename, local_files_only=True)
    except LocalEntryNotFoundError:
        return False
    return True


def ensure_model(model_name: str, on_progress=None) -> None:
    """Download the Demucs weights on first use and report 0..1 progress.

    Later calls find the file in the Hugging Face cache and return immediately.
    """
    spec = next(item for item in MODES.values() if item["model"] == model_name)
    label = spec["label"]
    size_mb = int(spec["download_mb"])

    def report(fraction: float, message: str) -> None:
        if on_progress is not None:
            on_progress(fraction, message)

    from demucs.hf import DEFAULT_NAMESPACE, hf_repo_name
    from huggingface_hub import hf_hub_download
    import yaml

    repo_id = f"{DEFAULT_NAMESPACE}/{hf_repo_name(model_name)}"
    try:
        yaml_path = hf_hub_download(repo_id, f"{model_name}.yaml")
        with open(yaml_path, encoding="utf-8") as handle:
            bag = yaml.safe_load(handle)
        filenames = [f"{signature}.safetensors" for signature in bag["models"]]
    except Exception as exc:
        logger.warning("Could not read the model list for %s: %s", model_name, exc)
        raise UserFacingError(
            f"{label}音源分离模型下载失败。第一次使用需要联网下载约 {size_mb}MB。"
            "请检查网络后重试。"
        ) from exc

    missing = [name for name in filenames if not _weights_cached(repo_id, name)]
    if not missing:
        report(1.0, f"{label}模型已经在这台电脑上，不用再下载。")
        return

    report(0.0, f"正在下载{label}模型，大约 {size_mb}MB…")
    bar = _download_bar(on_progress, label, size_mb)
    try:
        for filename in missing:
            hf_hub_download(repo_id, filename, tqdm_class=bar)
    except Exception as exc:
        logger.warning("Model download failed for %s: %s", model_name, exc)
        raise UserFacingError(
            f"{label}音源分离模型下载失败。第一次使用需要联网下载约 {size_mb}MB。"
            "请检查网络后，关掉程序再打开重试。"
        ) from exc
    report(1.0, f"{label}模型下载完成。")


def get_separator(model_name: str, on_progress=None):
    """Load one Demucs model and reuse it for later songs."""
    with _separator_lock:
        separator = _separators.get(model_name)
        if separator is None:
            ensure_model(model_name, on_progress)
            if on_progress is not None:
                label = next(item["label"] for item in MODES.values() if item["model"] == model_name)
                on_progress(1.0, f"正在把{label}模型读进内存…")
            try:
                from demucs.api import Separator
            except Exception as exc:
                raise UserFacingError(
                    "音源分离组件没有装好。请关掉程序，重新运行 start 脚本。"
                ) from exc
            try:
                separator = Separator(
                    model=model_name,
                    device=pick_device(),
                    shifts=1,
                    overlap=0.25,
                    split=True,
                    segment=None,
                    jobs=0,
                    progress=False,
                )
            except Exception as exc:
                spec = next(item for item in MODES.values() if item["model"] == model_name)
                raise UserFacingError(
                    f"{spec['label']}音源分离模型加载失败。第一次使用需要联网下载约 {spec['download_mb']}MB。"
                    "请检查网络后，关掉程序再打开重试。"
                ) from exc
            _separators[model_name] = separator
        return separator


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


def separate(path: Path, mode: str = "fine", device: str | None = None, on_chunk=None, on_download=None) -> tuple[dict, int]:
    """Return stem tensors keyed by name, plus the model's sample rate.

    Tensors are shaped (channels, frames) on CPU.
    """
    spec = mode_spec(mode)
    separator = get_separator(spec["model"], on_progress=on_download)
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
