"""BS-RoFormer models used by 「最高质量」.

The six-stem checkpoint is much cleaner than htdemucs_6s on drums, bass,
piano, and guitar. Its vocal stem stays silent on a choir pad, so vocals
come from htdemucs_6s. A second checkpoint then lifts bowed strings out of
what remains. 「其他乐器」 is whatever is still left in the mix.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from pathlib import Path

import numpy as np

from app.errors import UserFacingError

logger = logging.getLogger(__name__)

CKPT_REPO = "enerjazzer/BS-ROFO-SW-Fixed"
CKPT_FILE = "BS-Rofo-SW-Fixed.ckpt"
CKPT_SHA256 = "24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e"
CKPT_MB = 699
CONFIG_PATH = Path(__file__).resolve().parent / "model_configs" / "bs_roformer_sw.yaml"

STRINGS_REPO = "oulianov/BS-Roformer-BowedStrings-Duality"
STRINGS_FILE = "gilliaan_bowedstrings_bs_v1.ckpt"
STRINGS_SHA256 = "282fabc28fb106edcc5e5e8383ea36559602fe53d0c406c736e08f58d66710fc"
STRINGS_MB = 303
STRINGS_CONFIG = Path(__file__).resolve().parent / "model_configs" / "bs_roformer_bowed_strings.yaml"

_model = None
_config = None
_lock = threading.Lock()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_bar(on_progress, label: str = "最高质量", size_mb: int = CKPT_MB):
    from tqdm import tqdm

    class DownloadBar(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["file"] = open("/dev/null", "w")  # noqa: SIM115 — closed in close()
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            shown = super().update(n)
            total = float(self.total or 0)
            if total > 0 and on_progress is not None:
                on_progress(
                    min(1.0, self.n / total),
                    f"正在下载{label}模型… {self.n / 1e6:.0f}/{total / 1e6:.0f} MB。"
                    f"大约 {size_mb}MB，下完留在这台电脑上，下次不用再下。",
                )
            return shown

        def close(self):
            sink = getattr(self, "fp", None)
            super().close()
            if sink is not None and hasattr(sink, "close"):
                sink.close()

    return DownloadBar


def roformer_cached() -> bool:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        path = Path(hf_hub_download(CKPT_REPO, CKPT_FILE, local_files_only=True))
    except LocalEntryNotFoundError:
        return False
    return path.is_file() and path.stat().st_size > 1_000_000


def ensure_roformer(on_progress=None) -> Path:
    """Download the checkpoint on first use. Later calls find the cache."""
    from huggingface_hub import hf_hub_download

    def report(fraction: float, message: str) -> None:
        if on_progress is not None:
            on_progress(fraction, message)

    if roformer_cached():
        report(1.0, "最高质量模型已经在这台电脑上，不用再下载。")
        return Path(hf_hub_download(CKPT_REPO, CKPT_FILE, local_files_only=True))

    report(0.0, f"正在下载最高质量模型，大约 {CKPT_MB}MB…")
    try:
        path = Path(
            hf_hub_download(CKPT_REPO, CKPT_FILE, tqdm_class=_download_bar(on_progress, "最高质量", CKPT_MB))
        )
    except Exception as exc:
        logger.warning("RoFormer download failed: %s", exc)
        raise UserFacingError(
            f"最高质量音源分离模型下载失败。第一次使用需要联网下载约 {CKPT_MB}MB。"
            "请检查网络后，关掉程序再打开重试。"
        ) from exc
    digest = _sha256(path)
    if digest != CKPT_SHA256:
        path.unlink(missing_ok=True)
        raise UserFacingError("最高质量模型下载不完整。请检查网络后重试，程序会重新下载。")
    report(1.0, "最高质量模型下载完成。")
    return path


def strings_cached() -> bool:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        path = Path(hf_hub_download(STRINGS_REPO, STRINGS_FILE, local_files_only=True))
    except LocalEntryNotFoundError:
        return False
    return path.is_file() and path.stat().st_size > 1_000_000


def ensure_strings(on_progress=None) -> Path:
    """Download the bowed-strings checkpoint. Later calls find the cache."""
    from huggingface_hub import hf_hub_download

    def report(fraction: float, message: str) -> None:
        if on_progress is not None:
            on_progress(fraction, message)

    if strings_cached():
        report(1.0, "弦乐模型已经在这台电脑上，不用再下载。")
        return Path(hf_hub_download(STRINGS_REPO, STRINGS_FILE, local_files_only=True))

    report(0.0, f"正在下载弦乐模型，大约 {STRINGS_MB}MB…")
    try:
        path = Path(
            hf_hub_download(
                STRINGS_REPO, STRINGS_FILE, tqdm_class=_download_bar(on_progress, "弦乐", STRINGS_MB)
            )
        )
    except Exception as exc:
        logger.warning("Strings download failed: %s", exc)
        raise UserFacingError(
            f"弦乐分离模型下载失败。第一次使用需要联网下载约 {STRINGS_MB}MB。"
            "请检查网络后，关掉程序再打开重试。"
        ) from exc
    digest = _sha256(path)
    if digest != STRINGS_SHA256:
        path.unlink(missing_ok=True)
        raise UserFacingError("弦乐模型下载不完整。请检查网络后重试，程序会重新下载。")
    report(1.0, "弦乐模型下载完成。")
    return path


def release_roformer() -> None:
    """Drop the six-stem network so the strings model can use that memory."""
    global _model, _config
    import gc

    with _lock:
        model = _model
        _model = None
        _config = None
    if model is not None:
        try:
            model.to("cpu")
        except Exception:
            pass
        del model
    gc.collect()


def _load(on_progress=None):
    global _model, _config
    with _lock:
        if _model is not None:
            return _model, _config
        if on_progress is not None:
            on_progress(1.0, "正在把最高质量模型读进内存…")
        import torch
        import yaml
        from bs_roformer.inference import SafeLoaderWithTuple
        from bs_roformer.utils import get_model_from_config
        from ml_collections import ConfigDict

        weights = ensure_roformer(on_progress)
        try:
            with CONFIG_PATH.open(encoding="utf-8") as handle:
                config = ConfigDict(yaml.load(handle, Loader=SafeLoaderWithTuple))
            model = get_model_from_config("bs_roformer", config)
            state = torch.load(weights, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
            model.eval()
        except UserFacingError:
            raise
        except Exception as exc:
            logger.warning("RoFormer load failed: %s", exc)
            raise UserFacingError(
                "最高质量音源分离模型加载失败。请关掉程序，重新运行 install.sh 后再试。"
            ) from exc
        _model = model
        _config = config
        return model, config


def _window(window_size: int, fade_size: int, device: str):
    import torch

    fade_in = torch.linspace(0, 1, fade_size)
    fade_out = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fade_out
    window[:fade_size] *= fade_in
    return window.to(device)


def _remain_phrase(seconds: float) -> str:
    if seconds < 50:
        return "不到一分钟"
    minutes = max(1, int(round(seconds / 60)))
    return f"大约还要 {minutes} 分钟"


def _prepare_mix(audio: np.ndarray, sample_rate: int):
    import librosa

    array = np.asarray(audio, dtype=np.float32)
    if sample_rate != 44100:
        array = librosa.resample(array, orig_sr=sample_rate, target_sr=44100, axis=-1).astype(np.float32)
    if array.ndim == 1:
        array = np.stack([array, array], axis=0)
    return array


def _demix(
    model,
    config,
    audio: np.ndarray,
    on_chunk,
    fraction_lo: float,
    fraction_hi: float,
    headline: str,
) -> dict[str, np.ndarray]:
    """Overlap-add one loaded BS-RoFormer. Progress stays inside the fraction span."""
    import torch

    from app.separation import pick_device

    device = pick_device()
    model.to(device)
    mix = torch.tensor(np.asarray(audio, dtype=np.float32))
    chunk = int(config.inference.chunk_size)
    overlap = max(1, int(config.inference.num_overlap))
    step = max(1, chunk // overlap)
    fade = max(1, chunk // 10)
    border = chunk - step
    padded = False
    if mix.shape[1] > 2 * border and border > 0:
        mix = torch.nn.functional.pad(mix, (border, border), mode="reflect")
        padded = True

    windowing = _window(chunk, fade, device)
    names = list(config.training.instruments)
    mix = mix.to(device)
    shape = (len(names),) + tuple(mix.shape)
    result = torch.zeros(shape, dtype=torch.float32, device=device)
    counter = torch.zeros(shape, dtype=torch.float32, device=device)
    total = int(mix.shape[1])
    chunk_count = max(1, (total + step - 1) // step)
    started = time.perf_counter()
    index = 0
    done = 0
    with torch.inference_mode():
        while index < total:
            part = mix[:, index : index + chunk]
            length = int(part.shape[-1])
            if length < chunk:
                if length > chunk // 2 + 1:
                    part = torch.nn.functional.pad(part, (0, chunk - length), mode="reflect")
                else:
                    part = torch.nn.functional.pad(part, (0, chunk - length))
            estimate = model(part.unsqueeze(0))[0]
            window = windowing.clone()
            if index == 0:
                window[:fade] = 1
            elif index + chunk >= total:
                window[-fade:] = 1
            result[..., index : index + length] += estimate[..., :length] * window[..., :length]
            counter[..., index : index + length] += window[..., :length]
            index += step
            done += 1
            if on_chunk is not None:
                elapsed = time.perf_counter() - started
                remain = elapsed / done * (chunk_count - done)
                extra = "" if done == 1 else f"按刚才的速度，这一步{_remain_phrase(remain)}。"
                on_chunk(
                    {
                        "state": "end",
                        "fraction": fraction_lo + (fraction_hi - fraction_lo) * done / chunk_count,
                        "message": f"{headline}第 {done}/{chunk_count} 段。{extra}页面可以开着。",
                    }
                )
    estimated = (result / counter.clamp_min(1e-8)).detach().cpu().numpy()
    np.nan_to_num(estimated, copy=False, nan=0.0)
    if padded:
        estimated = estimated[..., border:-border]
    return {name: np.asarray(stem, dtype=np.float32) for name, stem in zip(names, estimated)}


def separate_roformer(audio: np.ndarray, sample_rate: int, on_chunk=None) -> dict[str, np.ndarray]:
    """Return stem arrays shaped (channels, frames) at 44.1 kHz."""
    mix = _prepare_mix(audio, sample_rate)
    model, config = _load()
    return _demix(
        model,
        config,
        mix,
        on_chunk,
        0.0,
        0.48,
        "正在分离音轨（最高质量）… ",
    )


def separate_strings(audio: np.ndarray, sample_rate: int, on_chunk=None) -> np.ndarray | None:
    """Return the bowed-strings stem, or None if that model cannot run.

    The network is dropped before returning so it does not sit next to the
    six-stem model in memory.
    """
    import torch
    import yaml
    from bs_roformer.inference import SafeLoaderWithTuple
    from bs_roformer.utils import get_model_from_config
    from ml_collections import ConfigDict

    mix = _prepare_mix(audio, sample_rate)
    model = None
    try:
        weights = ensure_strings()
        with STRINGS_CONFIG.open(encoding="utf-8") as handle:
            config = ConfigDict(yaml.load(handle, Loader=SafeLoaderWithTuple))
        if not hasattr(config.inference, "chunk_size"):
            config.inference.chunk_size = int(config.audio.chunk_size)
        model = get_model_from_config("bs_roformer", config)
        state = torch.load(weights, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        model.eval()
        stems = _demix(model, config, mix, on_chunk, 0.62, 0.98, "正在把弦乐拆出来… ")
        return stems["strings"]
    except Exception:
        logger.exception("弦乐模型没有跑起来，弦乐仍留在其他乐器里。")
        if on_chunk is not None:
            on_chunk(
                {
                    "state": "end",
                    "fraction": 0.98,
                    "message": "弦乐模型没有跑起来，小提琴和大提琴仍记在「其他乐器」里。",
                }
            )
        return None
    finally:
        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                pass
            del model
        import gc

        gc.collect()


def assemble_quality_stems(
    mix: np.ndarray,
    roformer: dict[str, np.ndarray],
    vocals: np.ndarray,
    strings: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Keep the RoFormer instrument stems and rebuild 「其他」 from the mix.

    Vocals are supplied by the caller. On the measured choir clip the RoFormer
    vocal stem is silent, and the voice would otherwise stay inside 「其他」.
    When a strings stem is supplied it is removed from that remainder too.
    """
    needed = ("drums", "bass", "piano", "guitar")
    arrays = [mix, vocals, *[roformer[name] for name in needed]]
    if strings is not None:
        arrays.append(strings)
    width = min(int(item.shape[-1]) for item in arrays)

    def cut(audio: np.ndarray) -> np.ndarray:
        array = np.asarray(audio, dtype=np.float32)
        if array.ndim == 1:
            array = np.stack([array, array], axis=0)
        return array[..., :width]

    stems = {
        "vocals": cut(vocals),
        "drums": cut(roformer["drums"]),
        "bass": cut(roformer["bass"]),
        "piano": cut(roformer["piano"]),
        "guitar": cut(roformer["guitar"]),
    }
    residual = cut(mix)
    for name in ("vocals", "drums", "bass", "piano", "guitar"):
        residual = residual - stems[name]
    if strings is not None:
        stems["strings"] = cut(strings)
        residual = residual - stems["strings"]
    stems["other"] = residual.astype(np.float32)
    return stems
