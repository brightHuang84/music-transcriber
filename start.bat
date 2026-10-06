@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 听音识谱：正在准备环境，第一次会比较久…

where python >nul 2>&1
if errorlevel 1 (
  echo 没有找到 Python。请安装 Python 3.12，并勾选 Add python.exe to PATH。
  echo 下载页面：https://www.python.org/downloads/
  pause
  exit /b 1
)

python -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)"
if errorlevel 1 (
  echo 需要 Python 3.10、3.11 或 3.12。请不要用 3.13 或更新的版本。
  pause
  exit /b 1
)

where ffmpeg >nul 2>&1
if errorlevel 1 (
  echo 没有找到 ffmpeg。请先打开 PowerShell，运行：
  echo     winget install Gyan.FFmpeg
  echo 安装后关掉并重新打开 start.bat。
  pause
  exit /b 1
)

if not exist .venv (
  python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install -U pip
if errorlevel 1 goto :failed

nvidia-smi >nul 2>&1
if errorlevel 1 (
  python -m pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu
) else (
  python -m pip install torch==2.4.1 torchaudio==2.4.1
)
if errorlevel 1 goto :failed

python -m pip install -r requirements.txt
if errorlevel 1 goto :failed
python -m pip install basic-pitch==0.4.0 --no-deps
if errorlevel 1 goto :failed

python -m app
if errorlevel 1 goto :failed
exit /b 0

:failed
echo.
echo 启动失败。请把上面的英文错误信息保留下来，方便排查。
pause
exit /b 1
