# 听音识谱

把一首歌放进去，程序会在**你自己的电脑**上：

1. 把歌分成四条音轨：人声、鼓、贝斯、其他乐器，并且可以单独听。
2. 给人声、贝斯、其他乐器识别出音符（音高、什么时候开始、持续多久、力度），画成钢琴卷帘，并标出 C4、E4 这样的音名。
3. 给鼓识别底鼓、军鼓、踩镲，画成对齐拍子的格子。
4. 估计整首歌的速度（BPM）、拍子和小节第一拍。能判断的话，也会标出简单的大和弦 / 小和弦。
5. 导出每一轨的 MIDI、合并后的 MIDI，以及分开的音频。

歌曲不会上传到网上。

## 你需要先安装两样东西

1. **Python 3.12**（3.10 或 3.11 也可以，不要用 3.13）
   - Windows：打开 [python.org/downloads](https://www.python.org/downloads/)，安装时勾选 **Add python.exe to PATH**。
   - macOS：同样可以从 python.org 下载，或者在终端运行 `brew install python@3.12`。
2. **ffmpeg**（用来读取 mp3、m4a、flac）
   - Windows：打开 PowerShell，运行 `winget install Gyan.FFmpeg`。装完后**关掉并重新打开**终端。
   - macOS：终端运行 `brew install ffmpeg`。
   - 如果没有 winget / brew，到 [ffmpeg.org/download.html](https://ffmpeg.org/download.html) 下载，并保证终端里输入 `ffmpeg` 有反应。

## 怎么打开

**Windows：** 双击 `start.bat`。

**macOS：** 在终端进入这个文件夹，运行：

```bash
chmod +x start.sh
./start.sh
```

第一次运行会创建虚拟环境并下载依赖，可能要十几分钟，取决于网速。以后再打开就会快很多。

看到「听音识谱已启动」后，浏览器会打开一个页面。如果没有自动打开，窗口里有一个地址，一般是 `http://127.0.0.1:8765`，把它复制到浏览器即可。

**不要关掉那个黑色窗口。** 关掉它，程序就退出了。

## 怎么用

1. 把 mp3、wav、m4a 或 flac 拖进页面，或点击「选择歌曲」。
2. 等待进度走完。分离音轨最慢。一首 3 分钟左右的歌，在没有独立显卡的电脑上常常要几分钟到十几分钟。
3. 完成后：
   - 点「人声 / 鼓 / 贝斯 / 其他乐器」切换你正在看的乐器。
   - 「原曲」边听整首歌边看当前乐器；「只听当前乐器」只放这一轨；「四轨一起」把分开的音轨叠回去。
   - 空格键可以播放 / 暂停。
   - 钢琴卷帘上的字就是音名。左边 C4 的 4 表示八度，数字越大音越高。
   - 鼓的格子按十六分音符对齐。竖线是拍，颜色更深的是小节第一拍。
4. 右侧可以下载当前音轨、当前 MIDI、全部 MIDI，或打包下载。

没有现成歌曲时，可以点「先分析一段 4 秒示例」。那是程序生成的小旋律，用来熟悉界面，同样会走完整分析。

## 结果可以怎么理解

- **C4** 是中央 C。唱名 do re mi fa sol la si 对应 C D E F G A B。
- 速度 **120 BPM** 表示一分钟大约 120 拍。
- 调性和和弦是自动估计的，只作参考。程序只标大和弦和小和弦。
- 自动识谱会有错音、漏音，八度也可能偏高或偏低。把它当成学习的起点，对照着耳朵听。

MIDI 可以用 [MuseScore](https://musescore.org/)（免费）打开，看更接近五线谱的样子。音轨名称是英文：Vocals、Bass、Other、Drums。

## 它在背后做了什么

- 音轨分离：[Demucs](https://github.com/facebookresearch/demucs) 的 **htdemucs** 模型（人声、鼓、贝斯、其他）。
- 音符识别：[Spotify Basic Pitch](https://github.com/spotify/basic-pitch)（ONNX，不需要安装 TensorFlow）。
- 速度、拍子、和弦：librosa，加上按频谱区分底鼓 / 军鼓 / 踩镲。
- 有 NVIDIA 显卡（并且 start 脚本能看到 `nvidia-smi`）或苹果芯片时，会尽量用显卡；否则用 CPU。CPU 一定可以跑。

第一次分离歌曲时，还会下载大约 80MB 的 Demucs 模型。

## 常见问题

**窗口一闪就关了，或提示找不到 Python。**  
重新安装 Python 3.12，并勾选加入 PATH。然后重新双击 `start.bat`。

**提示找不到 ffmpeg。**  
按上面的方法安装，然后关掉旧窗口，重新运行 start 脚本。

**一直停在「分离音轨」。**  
这是正常的，CPU 上这一步很慢。请让窗口开着。如果超过半小时仍是 0%，看黑色窗口里有没有红色报错。第一次失败常常是因为没联网，模型没下载下来。

**提示内存不够。**  
关掉其他占内存的软件，或把歌曲剪到 1–3 分钟再试。

**人声那一轨是空的。**  
可能是纯音乐，人声被分到了「其他乐器」。点开其他几轨看看。

**鼓点或和弦和我听到的不一样。**  
鼓的分类靠频谱，爵士鼓、电子鼓和很密的踩镲容易标错。和弦遇到七和弦、转位、很多乐器一起响时也会偏。旋律相对更可信，但仍然不是人工记谱。

**页面打不开。**  
确认黑色窗口还在。手动打开窗口里打印的地址。地址以 `127.0.0.1` 开头，表示只给你自己用。

## 给想改代码的人

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install "basic-pitch==0.4.0" --no-deps
# 先按 start.sh 里的方式装好 torch / torchaudio
pytest tests/test_analysis.py
pytest tests/test_smoke.py
```

`tests/test_smoke.py` 会用一段 4 秒合成音乐跑通分离、识谱、鼓点、MIDI。第一次会下载模型。

## 已知限制

- 默认假设 4/4 拍。三拍子的歌，小节线可能对不齐。
- 不识别歌词，也不生成五线谱图片（MIDI 可以交给 MuseScore）。
- 一首歌最长接受约 10 分钟，文件最大约 120MB。
- 鼓点分类不是专门的鼓转录模型，复杂鼓组只会标成底鼓、军鼓、踩镲或其他。
- 和弦只估计大三和弦和小三和弦。
- Python 3.13 及以上还不能用这一套依赖。
