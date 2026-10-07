#!/usr/bin/env bash
# 在 Ubuntu 上安装听音识谱：系统组件、Python 3.12、菜单图标。
# 装好后从应用程序菜单打开，不需要浏览器，也不会留下终端窗口。
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

echo "听音识谱：开始安装"

if ! command -v sudo >/dev/null 2>&1; then
  echo "没有找到 sudo。请用有管理员权限的账户运行这个脚本。"
  exit 1
fi
# sudo -v 会刷新密码缓存。已经配置成免密码的机器上，它仍可能因为没有终端而失败。
if sudo -n true 2>/dev/null; then
  echo "已有管理员权限，继续安装系统组件。"
else
  echo "安装过程会请你输入开机密码，用来安装 ffmpeg 等系统组件。"
  echo
  if ! sudo -v; then
    echo "没有拿到管理员权限，安装停下来了。"
    exit 1
  fi
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
    candidate="$(apt-cache policy "$package" 2>/dev/null | awk '/Candidate:/ {print $2; exit}')"
    if [ -n "${candidate}" ] && [ "${candidate}" != "(none)" ]; then
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

# uv 自己下载的 Python。系统里即使有 python3.12，也不要用它来建环境。
# uv 新建的虚拟环境默认没有 pip，所以后面一律用 uv pip，不调用 python -m pip。
export UV_PYTHON_PREFERENCE=only-managed

echo "正在准备 Python 3.12…"
uv python install 3.12

PY="${ROOT}/.venv/bin/python"
if [ -e "${ROOT}/.venv" ] && [ ! -x "${PY}" ]; then
  echo "已有的虚拟环境不完整，将重建。"
  rm -rf "${ROOT}/.venv"
elif [ -x "${PY}" ] && ! "${PY}" -c 'import sys; raise SystemExit(0 if sys.version_info[:2]==(3, 12) else 1)'; then
  echo "已有的虚拟环境不是 Python 3.12，将重建。"
  rm -rf "${ROOT}/.venv"
fi
if [ ! -x "${PY}" ]; then
  uv venv --python 3.12 "${ROOT}/.venv"
fi

pyinstall() {
  uv pip install --python "${PY}" "$@"
}

if ! "${PY}" -c "import torch" >/dev/null 2>&1; then
  echo "正在安装 PyTorch…"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "检测到 NVIDIA 显卡，安装 CUDA 版。若失败会改用 CPU 版。"
    if ! pyinstall "torch==2.4.1" "torchaudio==2.4.1"; then
      echo "显卡版没有装上，改为 CPU 版。分析仍然可以跑，只是更慢。"
      pyinstall "torch==2.4.1" "torchaudio==2.4.1" --index-url https://download.pytorch.org/whl/cpu
    fi
  else
    echo "没有检测到 NVIDIA 显卡，安装 CPU 版。"
    pyinstall "torch==2.4.1" "torchaudio==2.4.1" --index-url https://download.pytorch.org/whl/cpu
  fi
fi

echo "正在安装分析和窗口组件（第一次会比较久）…"
pyinstall -r "${ROOT}/requirements.txt"
pyinstall "basic-pitch==0.4.0" --no-deps
pyinstall -r "${ROOT}/requirements-desktop.txt"

echo "正在加入应用程序菜单…"
"${PY}" -m app.install_desktop --root "${ROOT}"

echo
echo "安装完成。"
echo "打开方式：按 Super 键（Windows 键），搜索「听音识谱」，点击图标。"
echo "不需要打开浏览器，也不会出现终端窗口。"
echo "若想卸掉菜单项和虚拟环境，运行 ./uninstall.sh"
