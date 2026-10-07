"""Register 听音识谱 in the desktop application menu, and remove it again."""

from __future__ import annotations

import argparse
import importlib.util
import shlex
import shutil
import stat
import subprocess
import sys
from pathlib import Path

APP_ID = "tingyin-shipu"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def desktop_file_path() -> Path:
    return Path.home() / ".local" / "share" / "applications" / f"{APP_ID}.desktop"


def installed_icon_path() -> Path:
    return Path.home() / ".local" / "share" / "icons" / "hicolor" / "256x256" / "apps" / f"{APP_ID}.png"


def launcher_path(root: Path) -> Path:
    return root / "bin" / APP_ID


def desktop_entry_text(exec_path: Path, icon_path: Path) -> str:
    """A menu entry that opens a window and does not open a terminal."""
    return "\n".join(
        [
            "[Desktop Entry]",
            "Version=1.0",
            "Type=Application",
            "Name=听音识谱",
            "GenericName=音乐分析",
            "Comment=把歌曲拆开，看见每个乐器的音符和节奏",
            f"Exec={shlex.quote(str(exec_path))}",
            f"Icon={shlex.quote(str(icon_path))}",
            "Terminal=false",
            "Categories=AudioVideo;Music;",
            "Keywords=music;midi;audio;谱;",
            "StartupNotify=true",
            f"StartupWMClass={APP_ID}",
            "",
        ]
    )


def render_launcher(root: Path) -> str:
    python = root / ".venv" / "bin" / "python"
    log_path = Path.home() / ".local" / "share" / APP_ID / "app.log"
    root_q = shlex.quote(str(root))
    python_q = shlex.quote(str(python))
    log_q = shlex.quote(str(log_path))
    log_dir_q = shlex.quote(str(log_path.parent))
    return f"""#!/bin/bash
set -euo pipefail
cd {root_q}
mkdir -p {log_dir_q}
export PYTHONNOUSERSITE=1
if ! {python_q} -m app.desktop >>{log_q} 2>&1; then
  if command -v zenity >/dev/null 2>&1; then
    zenity --error --width=420 --title="听音识谱" \\
      --text="听音识谱没有打开。\\n请把这个文件发给懂电脑的朋友：\\n{log_path}"
  fi
  exit 1
fi
"""


def ensure_icon(root: Path) -> Path:
    icon = root / "share" / f"{APP_ID}.png"
    if icon.exists():
        return icon
    maker = root / "share" / "make_icon.py"
    spec = importlib.util.spec_from_file_location("tingyin_make_icon", maker)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法生成图标：{maker}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    icon.parent.mkdir(parents=True, exist_ok=True)
    icon.write_bytes(module.render())
    return icon


def install_user_entry(root: Path | None = None) -> dict[str, Path]:
    root = (root or repo_root()).resolve()
    icon_src = ensure_icon(root)
    launcher = launcher_path(root)
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text(render_launcher(root), encoding="utf-8")
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    icon_dst = installed_icon_path()
    icon_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(icon_src, icon_dst)

    desktop = desktop_file_path()
    desktop.parent.mkdir(parents=True, exist_ok=True)
    desktop.write_text(desktop_entry_text(launcher, icon_dst), encoding="utf-8")

    app_dir = desktop.parent
    if shutil.which("update-desktop-database"):
        subprocess.run(["update-desktop-database", str(app_dir)], check=False, capture_output=True)
    return {"launcher": launcher, "desktop": desktop, "icon": icon_dst}


def uninstall_user_entry() -> list[Path]:
    removed: list[Path] = []
    for path in (desktop_file_path(), installed_icon_path()):
        if path.exists():
            path.unlink()
            removed.append(path)
    app_dir = desktop_file_path().parent
    if shutil.which("update-desktop-database") and app_dir.exists():
        subprocess.run(["update-desktop-database", str(app_dir)], check=False, capture_output=True)
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="注册或移除听音识谱的应用程序菜单项")
    parser.add_argument("--remove", action="store_true")
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.remove:
        removed = uninstall_user_entry()
        if removed:
            print("已从应用程序菜单移除：")
            for path in removed:
                print(f"  {path}")
        else:
            print("菜单里没有听音识谱，不用移除。")
        return 0
    paths = install_user_entry(args.root)
    print("已加入应用程序菜单。在活动概览里搜索「听音识谱」即可打开。")
    print(paths["desktop"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
