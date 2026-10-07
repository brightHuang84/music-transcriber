#!/usr/bin/env bash
# 从应用程序菜单移除听音识谱，并删除本目录里的虚拟环境。
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

if [ -x "${ROOT}/.venv/bin/python" ]; then
  "${ROOT}/.venv/bin/python" -m app.install_desktop --remove
else
  rm -f "${HOME}/.local/share/applications/tingyin-shipu.desktop"
  rm -f "${HOME}/.local/share/icons/hicolor/256x256/apps/tingyin-shipu.png"
  if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "${HOME}/.local/share/applications" || true
  fi
  echo "已尝试从应用程序菜单移除听音识谱。"
fi

rm -rf "${ROOT}/.venv" "${ROOT}/bin/tingyin-shipu"
echo "虚拟环境已删除。"
echo "项目文件夹还在。如果不再需要这个软件，把整个文件夹删掉即可。"
