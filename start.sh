#!/usr/bin/env bash
# 双击或在终端运行：自动准备环境并打开浏览器。
set -euo pipefail
cd "$(dirname "$0")"

echo "听音识谱：正在准备环境，第一次会比较久…"

pick_python() {
  local candidate version
  for candidate in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      version="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
      case "$version" in
        3.10|3.11|3.12) echo "$candidate"; return 0 ;;
      esac
    fi
  done
  return 1
}

if ! PY="$(pick_python)"; then
  echo "没有找到 Python 3.10、3.11 或 3.12。"
  echo "请先安装 Python 3.12：https://www.python.org/downloads/"
  echo "macOS 也可以在终端运行：brew install python@3.12"
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "没有找到 ffmpeg，还不能读取 mp3 / m4a。"
  echo "macOS：brew install ffmpeg"
  echo "Ubuntu / Debian：sudo apt install ffmpeg"
  echo "也可以到 https://ffmpeg.org/download.html 下载。"
  exit 1
fi

if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip

OS="$(uname -s)"
if [ "$OS" = "Darwin" ]; then
  python -m pip install "torch==2.4.1" "torchaudio==2.4.1"
elif command -v nvidia-smi >/dev/null 2>&1; then
  python -m pip install "torch==2.4.1" "torchaudio==2.4.1"
else
  python -m pip install "torch==2.4.1" "torchaudio==2.4.1" --index-url https://download.pytorch.org/whl/cpu
fi

python -m pip install -r requirements.txt
python -m pip install "basic-pitch==0.4.0" --no-deps

exec python -m app
