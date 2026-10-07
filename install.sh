#!/usr/bin/env bash
# 在 Ubuntu 上安装听音识谱：系统组件、Python 3.12、菜单图标。
# 装好后从应用程序菜单打开，不需要浏览器，也不会留下终端窗口。
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

echo "听音识谱：开始安装"
echo "安装过程会请你输入开机密码，用来安装 ffmpeg 等系统组件。"
echo

if ! command -v sudo >/dev/null 2>&1; then
  echo "没有找到 sudo。请用有管理员权限的账户运行这个脚本。"
  exit 1
fi
if ! sudo -v; then
  echo "没有拿到管理员权限，安装停下来了。"
  exit 1
fi

if command -v apt-get >/dev/null 2>&1; then
  echo "正在安装系统组件（ffmpeg、中文字体、窗口所需的库）…"
  sudo apt-get update -qq
  packages=(ffmpeg ca-certificates curl zenity fonts-noto-cjk)
  candidates=(
    libnss3 libnspr4
    libxkbcommon0 libxkbcommon-x11-0
    libxcb-cursor0 libxcb-xinerama0 libxcb-icccm4 libxcb-image0
    libxcb-keysyms1 libxcb-render-util0 libxcb-shape0
    libxcomposite1 libxdamage1 libxrandr2 libxfixes3 libxi6 libxtst6
    libgbm1 libdrm2 libxshmfence1
    libpango-1.0-0 libcups2
    libatk1.0-0 libatk-bridge2.0-0
    libgtk-3-0t64 libgtk-3-0
    libgl1 libegl1 libopengl0
    libasound2t64 libasound2
  )
  for package in "${candidates[@]}"; do
    if apt-cache show "$package" >/dev/null 2>&1; then
      packages+=("$package")
    fi
  done
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "${packages[@]}"
else
  echo "这不是 apt 系统。请先自行安装 ffmpeg，再继续。"
  if ! command -v ffmpeg >/dev/null 2>&1; then
    exit 1
  fi
fi

export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
  echo "正在安装 uv，用来下载 Python 3.12（不使用系统自带的 Python）…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
  if [ -f "${HOME}/.local/bin/env" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/.local/bin/env"
  fi
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv 没有安装成功。请检查网络后重新运行 ./install.sh。"
  exit 1
fi

echo "正在准备 Python 3.12…"
uv python install 3.12
need_venv=0
if [ ! -x "${ROOT}/.venv/bin/python" ]; then
  need_venv=1
elif ! "${ROOT}/.venv/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info[:2]==(3, 12) else 1)'; then
  echo "已有的虚拟环境不是 Python 3.12，将重建。"
  rm -rf "${ROOT}/.venv"
  need_venv=1
fi
if [ "${need_venv}" -eq 1 ]; then
  uv venv --python 3.12 "${ROOT}/.venv"
fi

PY="${ROOT}/.venv/bin/python"
"${PY}" -m pip install -U pip

if ! "${PY}" -c "import torch" >/dev/null 2>&1; then
  echo "正在安装 PyTorch…"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "检测到 NVIDIA 显卡，安装 CUDA 版。若失败会改用 CPU 版。"
    if ! "${PY}" -m pip install "torch==2.4.1" "torchaudio==2.4.1"; then
      echo "显卡版没有装上，改为 CPU 版。分析仍然可以跑，只是更慢。"
      "${PY}" -m pip install "torch==2.4.1" "torchaudio==2.4.1" --index-url https://download.pytorch.org/whl/cpu
    fi
  else
    echo "没有检测到 NVIDIA 显卡，安装 CPU 版。"
    "${PY}" -m pip install "torch==2.4.1" "torchaudio==2.4.1" --index-url https://download.pytorch.org/whl/cpu
  fi
fi

echo "正在安装分析和窗口组件（第一次会比较久）…"
"${PY}" -m pip install -r "${ROOT}/requirements.txt"
"${PY}" -m pip install "basic-pitch==0.4.0" --no-deps
"${PY}" -m pip install -r "${ROOT}/requirements-desktop.txt"

echo "正在加入应用程序菜单…"
"${PY}" -m app.install_desktop --root "${ROOT}"

echo
echo "安装完成。"
echo "打开方式：按 Super 键（Windows 键），搜索「听音识谱」，点击图标。"
echo "不需要打开浏览器，也不会出现终端窗口。"
echo "若想卸掉菜单项和虚拟环境，运行 ./uninstall.sh"
