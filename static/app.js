const STEMS = {
  vocals: { label: "人声", en: "Vocals", pitched: true, color: "#d4534a" },
  drums: { label: "鼓", en: "Drums", pitched: false, color: "#c4892a" },
  bass: { label: "贝斯", en: "Bass", pitched: true, color: "#2d6d9a" },
  other: { label: "其他乐器", en: "Other", pitched: true, color: "#3d8b6e" },
};

const DRUM_ROWS = [
  { kind: "crash", label: "吊镲", en: "Crash" },
  { kind: "ride", label: "叮叮镲", en: "Ride" },
  { kind: "hat_open", label: "开镲", en: "Open hat" },
  { kind: "hat_closed", label: "闭镲", en: "Closed hat" },
  { kind: "tom_high", label: "高通鼓", en: "High tom" },
  { kind: "tom_mid", label: "中通鼓", en: "Mid tom" },
  { kind: "snare", label: "军鼓", en: "Snare" },
  { kind: "tom_floor", label: "落地通鼓", en: "Floor tom" },
  { kind: "kick", label: "底鼓", en: "Kick" },
];

const STEP_ORDER = ["read", "separate", "rhythm", "notes", "drums", "chords", "midi", "done"];

const dropView = document.querySelector("#drop-view");
const progressView = document.querySelector("#progress-view");
const errorView = document.querySelector("#error-view");
const resultView = document.querySelector("#result-view");
const dropzone = document.querySelector("#dropzone");
const fileInput = document.querySelector("#file-input");

let job = null;
let result = null;
let viewStem = "vocals";
let listenMode = "mix";
let audioCtx = null;
let buffers = {};
let playing = false;
let pauseOffset = 0;
let startedAt = 0;
let sources = [];
let raf = 0;
let pxPerSec = 80;
let playGeneration = 0;
let followPlayback = true;
let programmaticScroll = 0;

const DRUM_LABEL = Object.fromEntries(DRUM_ROWS.map((row) => [row.kind, row.label]));
const DRUM_EN = Object.fromEntries(DRUM_ROWS.map((row) => [row.kind, row.en]));

function show(name) {
  dropView.hidden = name !== "drop";
  progressView.hidden = name !== "progress";
  errorView.hidden = name !== "error";
  resultView.hidden = name !== "result";
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) seconds = 0;
  const whole = Math.floor(seconds);
  const minutes = Math.floor(whole / 60);
  const rest = whole % 60;
  return `${minutes}:${String(rest).padStart(2, "0")}`;
}

async function readError(response) {
  try {
    const body = await response.json();
    return body.detail || body.message || "请求没有成功。";
  } catch {
    return "请求没有成功。请确认程序还开着。";
  }
}

async function upload(file) {
  const name = (file.name || "").toLowerCase();
  if (!/\.(mp3|wav|m4a|flac)$/.test(name)) {
    showError("请选择 mp3、wav、m4a 或 flac 文件。");
    return;
  }
  const body = new FormData();
  body.append("file", file);
  showProgress(file.name, { status: "queued", progress: 0, step: "read", message: "正在上传…" });
  let response;
  try {
    response = await fetch("/api/analyze", { method: "POST", body });
  } catch {
    showError("连不上本机程序。请关掉听音识谱，再从应用程序菜单重新打开。");
    return;
  }
  if (!response.ok) {
    showError(await readError(response));
    return;
  }
  const data = await response.json();
  history.replaceState(null, "", `/?job=${data.id}`);
  poll(data.id);
}

async function startDemo() {
  showProgress("示例音乐.wav", { status: "queued", progress: 0, step: "read", message: "正在准备示例…" });
  let response;
  try {
    response = await fetch("/api/demo", { method: "POST" });
  } catch {
    showError("连不上本机程序。请关掉听音识谱，再从应用程序菜单重新打开。");
    return;
  }
  if (!response.ok) {
    showError(await readError(response));
    return;
  }
  const data = await response.json();
  history.replaceState(null, "", `/?job=${data.id}`);
  poll(data.id);
}

function showProgress(filename, data) {
  show("progress");
  document.querySelector("#progress-filename").textContent = filename || "正在分析";
  document.querySelector("#progress-message").textContent = data.message || "准备中…";
  document.querySelector("#progress-percent").textContent = `${data.progress || 0}%`;
  document.querySelector("#bar-fill").style.width = `${data.progress || 0}%`;
  const current = STEP_ORDER.indexOf(data.step || "read");
  document.querySelectorAll(".steps li").forEach((item) => {
    const index = STEP_ORDER.indexOf(item.dataset.step);
    item.classList.toggle("done", index < current);
    item.classList.toggle("current", index === current);
  });
}

function showError(message) {
  show("error");
  document.querySelector("#error-message").textContent = message;
}

async function poll(id) {
  let response;
  try {
    response = await fetch(`/api/jobs/${id}`);
  } catch {
    showError("连不上本机程序。请关掉听音识谱，再从应用程序菜单重新打开。");
    return;
  }
  if (!response.ok) {
    showError(await readError(response));
    return;
  }
  const data = await response.json();
  job = data;
  if (data.status === "done") {
    await present(data);
    return;
  }
  if (data.status === "error") {
    showError(data.message || "分析失败了。");
    return;
  }
  showProgress(data.filename, data);
  window.setTimeout(() => poll(id), 800);
}

function setScroll(element, property, value) {
  programmaticScroll += 1;
  element[property] = value;
  window.setTimeout(() => {
    programmaticScroll -= 1;
  }, 0);
}

function syncTransport() {
  const play = document.querySelector("#play");
  play.textContent = playing ? "暂停" : "播放";
  play.setAttribute("aria-label", playing ? "暂停" : "播放");
  const follow = document.querySelector("#follow");
  if (!follow) return;
  follow.textContent = followPlayback ? "跟随" : "恢复跟随";
  follow.classList.toggle("on", followPlayback);
  follow.setAttribute("aria-pressed", followPlayback ? "true" : "false");
}

function pauseFollow() {
  if (!followPlayback) return;
  followPlayback = false;
  syncTransport();
}

function userMovedScroll() {
  if (programmaticScroll > 0 || !playing) return;
  pauseFollow();
}

function stopPlayback() {
  playGeneration += 1;
  sources.forEach((source) => {
    try { source.stop(); } catch { /* already stopped */ }
  });
  sources = [];
  playing = false;
  cancelAnimationFrame(raf);
  syncTransport();
}

function stopToStart() {
  pauseOffset = 0;
  stopPlayback();
  paint(0);
  const scroller = document.querySelector("#roll-scroll");
  if (scroller) setScroll(scroller, "scrollLeft", 0);
}

function currentTime() {
  if (!playing || !audioCtx) return pauseOffset;
  return Math.min(result.duration, audioCtx.currentTime - startedAt);
}

function audibleNames() {
  if (listenMode === "mix") return ["mix"];
  if (listenMode === "all") return ["vocals", "drums", "bass", "other"];
  return [viewStem];
}

function restartIfPlaying() {
  if (!playing) return;
  const at = currentTime();
  stopPlayback();
  pauseOffset = at;
  startPlayback();
}

function startPlayback() {
  if (!audioCtx) audioCtx = new AudioContext();
  audioCtx.resume();
  const generation = ++playGeneration;
  sources = audibleNames().filter((name) => buffers[name]).map((name) => {
    const source = audioCtx.createBufferSource();
    source.buffer = buffers[name];
    source.connect(audioCtx.destination);
    source.start(0, pauseOffset);
    source.onended = () => {
      if (generation !== playGeneration) return;
      if (currentTime() >= result.duration - 0.08) {
        pauseOffset = 0;
        stopPlayback();
        paint(0);
      }
    };
    return source;
  });
  startedAt = audioCtx.currentTime - pauseOffset;
  playing = true;
  syncTransport();
  tick();
}

function togglePlay() {
  if (!result) return;
  if (playing) {
    pauseOffset = currentTime();
    stopPlayback();
    return;
  }
  startPlayback();
}

function seek(seconds) {
  pauseOffset = Math.min(result.duration, Math.max(0, seconds));
  if (playing) {
    stopPlayback();
    startPlayback();
  } else {
    paint(pauseOffset);
  }
}

function tick() {
  const time = currentTime();
  paint(time);
  if (playing) raf = requestAnimationFrame(tick);
}

function chordAt(time) {
  return (result.chords || []).find((chord) => time >= chord.start && time < chord.end);
}

function notesAt(time) {
  const stem = result.stems[viewStem];
  if (!stem) return [];
  if (viewStem === "drums") {
    return (stem.hits || [])
      .filter((hit) => Math.abs(hit.time - time) < 0.09)
      .map((hit) => ({ name: DRUM_LABEL[hit.kind] || hit.kind, solfege: hit.kind }));
  }
  if (!stem.notes) return [];
  return stem.notes.filter((note) => time >= note.start && time < note.end);
}

function paint(time) {
  document.querySelector("#time-now").textContent = formatTime(time);
  document.querySelector("#scrub").value = String(Math.round((time / result.duration) * 1000));
  const playhead = document.querySelector("#playhead");
  playhead.style.left = `${time * pxPerSec}px`;
  const scroller = document.querySelector("#roll-scroll");
  if (playing && followPlayback) {
    const x = time * pxPerSec;
    if (x < scroller.scrollLeft + 40 || x > scroller.scrollLeft + scroller.clientWidth - 60) {
      setScroll(scroller, "scrollLeft", Math.max(0, x - scroller.clientWidth * 0.3));
    }
  }
  const sounding = notesAt(time);
  document.querySelector("#now-notes").textContent = sounding.length
    ? sounding.map((note) => `${note.name}  ${note.solfege}`).join("   ")
    : "—";
  const chord = chordAt(time);
  document.querySelector("#now-chord").textContent = chord ? `${chord.symbol}  ${chord.name}` : "—";
  const beat = (result.beats || []).some((item) => Math.abs(item - time) < 0.08);
  document.querySelector("#beat-dot").classList.toggle("on", beat);
  document.querySelectorAll("#note-body tr").forEach((row) => {
    const start = Number(row.dataset.start);
    const end = Number(row.dataset.end);
    const active = time >= start && time < end;
    row.classList.toggle("active", active);
  });
  if (playing && followPlayback) {
    const activeRow = document.querySelector("#note-body tr.active");
    const box = document.querySelector(".table-scroll");
    if (activeRow && box) {
      const top = activeRow.offsetTop;
      const bottom = top + activeRow.offsetHeight;
      if (top < box.scrollTop + 4 || bottom > box.scrollTop + box.clientHeight - 4) {
        setScroll(box, "scrollTop", Math.max(0, top - box.clientHeight * 0.35));
      }
    }
  }
  if (viewStem === "drums") drawDrums(time);
  else drawPiano(time);
}

function layoutWidth() {
  const scroller = document.querySelector("#roll-scroll");
  const fit = scroller.clientWidth / Math.max(result.duration, 1);
  // Short clips fill the panel. Longer songs stay at least 70px per second and scroll.
  pxPerSec = Math.max(70, fit);
  const width = Math.max(scroller.clientWidth, Math.ceil(result.duration * pxPerSec) + 8);
  document.querySelector("#roll-inner").style.width = `${width}px`;
  return width;
}

function drawGrid(ctx, width, height, top) {
  const bpm = result.bpm || 120;
  const beat = 60 / bpm;
  ctx.save();
  ctx.strokeStyle = "rgba(36, 28, 22, 0.08)";
  ctx.lineWidth = 1;
  (result.beats || []).forEach((time) => {
    const x = time * pxPerSec;
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, height);
    ctx.stroke();
  });
  ctx.strokeStyle = "rgba(194, 75, 44, 0.35)";
  (result.downbeats || []).forEach((time) => {
    const x = time * pxPerSec;
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, height);
    ctx.stroke();
  });
  ctx.restore();
  return beat;
}

function drawPiano(time) {
  const canvas = document.querySelector("#viz");
  const width = layoutWidth();
  const height = 320;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const notes = result.stems[viewStem].notes || [];
  const gutter = document.querySelector("#gutter");
  gutter.innerHTML = "";
  gutter.style.height = `${height}px`;
  if (!notes.length) {
    ctx.fillStyle = "#6f6256";
    ctx.font = "16px sans-serif";
    ctx.fillText("这一轨没有识别到音符。", 16, 40);
    return;
  }
  const pitches = notes.map((note) => note.pitch);
  const minPitch = Math.min(...pitches) - 2;
  const maxPitch = Math.max(...pitches) + 2;
  const rows = Math.max(8, maxPitch - minPitch + 1);
  const rowH = height / rows;
  drawGrid(ctx, width, height, 0);
  for (let pitch = maxPitch; pitch >= minPitch; pitch -= 1) {
    const y = (maxPitch - pitch) * rowH;
    if ((maxPitch - pitch) % 2 === 0) {
      ctx.fillStyle = "rgba(36, 28, 22, 0.03)";
      ctx.fillRect(0, y, width, rowH);
    }
    if (pitch % 12 === 0) {
      const label = document.createElement("span");
      label.textContent = midiName(pitch);
      label.style.top = `${y + rowH / 2}px`;
      gutter.appendChild(label);
    }
  }
  const color = STEMS[viewStem].color;
  notes.forEach((note) => {
    const x = note.start * pxPerSec;
    const w = Math.max(4, (note.end - note.start) * pxPerSec - 2);
    const y = (maxPitch - note.pitch) * rowH + 2;
    const sounding = time >= note.start && time < note.end;
    ctx.globalAlpha = 0.45 + (note.velocity / 127) * 0.55;
    ctx.fillStyle = color;
    roundRect(ctx, x, y, w, rowH - 4, 5);
    ctx.fill();
    if (sounding) {
      ctx.globalAlpha = 1;
      ctx.strokeStyle = "#241c16";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    if (w > 36) {
      ctx.globalAlpha = 1;
      ctx.fillStyle = "#fffdf8";
      ctx.font = "12px sans-serif";
      ctx.fillText(note.name, x + 4, y + rowH / 2 + 1);
    }
  });
  ctx.globalAlpha = 1;
}

function drawDrums(time) {
  const canvas = document.querySelector("#viz");
  const width = layoutWidth();
  const height = Math.max(360, DRUM_ROWS.length * 40);
  const dpr = window.devicePixelRatio || 1;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const gutter = document.querySelector("#gutter");
  gutter.innerHTML = "";
  gutter.style.height = `${height}px`;
  const rowH = height / DRUM_ROWS.length;
  DRUM_ROWS.forEach((row, index) => {
    const label = document.createElement("span");
    label.textContent = row.label;
    label.style.top = `${index * rowH + rowH / 2}px`;
    gutter.appendChild(label);
    ctx.fillStyle = index % 2 ? "rgba(36, 28, 22, 0.03)" : "transparent";
    ctx.fillRect(0, index * rowH, width, rowH);
  });
  const bpm = result.bpm || 120;
  const step = (60 / bpm) / 4;
  const origin = (result.downbeats && result.downbeats[0]) || 0;
  let gridStart = origin;
  while (gridStart > 0) gridStart -= step;
  const hits = result.stems.drums.hits || [];
  drawGrid(ctx, width, height, 0);
  const currentIndex = Math.floor((time - gridStart) / step);
  for (let index = 0, x = gridStart; x < result.duration + step; index += 1, x += step) {
    if (index === currentIndex) {
      ctx.fillStyle = "rgba(194, 75, 44, 0.08)";
      ctx.fillRect(x * pxPerSec, 0, step * pxPerSec, height);
    }
  }
  hits.forEach((hit) => {
    const row = DRUM_ROWS.findIndex((item) => item.kind === hit.kind);
    if (row < 0) return;
    const index = Math.round((hit.time - gridStart) / step);
    const x = (gridStart + index * step) * pxPerSec + 2;
    const y = row * rowH + rowH * 0.22;
    const cell = Math.max(7, Math.min(16, step * pxPerSec * 0.62));
    ctx.globalAlpha = 0.55 + (hit.velocity / 127) * 0.45;
    ctx.fillStyle = STEMS.drums.color;
    roundRect(ctx, x, y, cell, rowH * 0.56, 4);
    ctx.fill();
  });
  ctx.globalAlpha = 1;
}

function midiName(midi) {
  const names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  return `${names[midi % 12]}${Math.floor(midi / 12) - 1}`;
}

function roundRect(ctx, x, y, w, h, r) {
  const radius = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + w, y, x + w, y + h, radius);
  ctx.arcTo(x + w, y + h, x, y + h, radius);
  ctx.arcTo(x, y + h, x, y, radius);
  ctx.arcTo(x, y, x + w, y, radius);
  ctx.closePath();
}

function renderChords() {
  const lane = document.querySelector("#chord-lane");
  lane.innerHTML = "";
  lane.style.width = `${layoutWidth()}px`;
  (result.chords || []).forEach((chord) => {
    const chip = document.createElement("div");
    chip.className = "chord-chip";
    chip.textContent = chord.symbol;
    chip.title = chord.name;
    chip.style.left = `${chord.start * pxPerSec}px`;
    chip.style.width = `${Math.max(18, (chord.end - chord.start) * pxPerSec - 4)}px`;
    lane.appendChild(chip);
  });
}

function renderNoteTable() {
  const body = document.querySelector("#note-body");
  const title = document.querySelector("#list-title");
  const hint = document.querySelector("#list-hint");
  body.innerHTML = "";
  if (viewStem === "drums") {
    title.textContent = "鼓点";
    const hits = result.stems.drums.hits || [];
    hits.slice(0, 400).forEach((hit) => {
      const row = document.createElement("tr");
      row.dataset.start = String(hit.time);
      row.dataset.end = String(hit.time + 0.1);
      row.innerHTML = `<td>${DRUM_LABEL[hit.kind] || hit.kind}</td><td>${DRUM_EN[hit.kind] || ""}</td><td>${formatTime(hit.time)}</td><td>—</td><td>${hit.velocity}</td>`;
      body.appendChild(row);
    });
    hint.textContent = hits.length > 400 ? "只列出前 400 个鼓点，上面的格子是完整的。" : `一共 ${hits.length} 下。`;
    return;
  }
  title.textContent = `${STEMS[viewStem].label}的音符`;
  const notes = result.stems[viewStem].notes || [];
  notes.slice(0, 400).forEach((note) => {
    const row = document.createElement("tr");
    row.dataset.start = String(note.start);
    row.dataset.end = String(note.end);
    row.innerHTML = `<td>${note.name}</td><td>${note.solfege}</td><td>${formatTime(note.start)}</td><td>${note.duration.toFixed(2)} 秒</td><td>${note.velocity}</td>`;
    body.appendChild(row);
  });
  hint.textContent = notes.length
    ? (notes.length > 400 ? "只列出前 400 个音，钢琴卷帘里是完整的。" : `一共 ${notes.length} 个音。点某一行不会跳转，播放到那里时会标出来。`)
    : "没有识别到音符。可能这一轨很安静，或者主旋律在另一条音轨里。";
}

function renderDownloads() {
  const grid = document.querySelector("#download-grid");
  const id = job.id;
  const items = [
    ["这一轨音频", `/api/jobs/${id}/audio/${viewStem}`, `${viewStem}.wav`],
    ["这一轨 MIDI", `/api/jobs/${id}/midi/${viewStem}`, `${viewStem}.mid`],
    ["原曲音频", `/api/jobs/${id}/audio/mix`, "mix.wav"],
    ["全部 MIDI", `/api/jobs/${id}/midi/all`, "all.mid"],
  ];
  grid.innerHTML = items.map(([label, href, filename]) =>
    `<a href="${href}" download="${filename}">${label}</a>`).join("");
  const bundle = document.querySelector("#bundle-link");
  bundle.href = `/api/jobs/${id}/bundle`;
  bundle.setAttribute("download", "music-analysis.zip");
}

function describeStem() {
  const stem = result.stems[viewStem];
  const meta = STEMS[viewStem];
  const note = document.querySelector("#stem-note");
  const hint = document.querySelector("#viz-hint");
  if (stem.silent) {
    note.textContent = `${meta.label}这一轨几乎没有声音，这首歌里可能没有它，或者它被分到了别的轨。`;
  } else if (viewStem === "drums") {
    const hits = stem.hits || [];
    const parts = DRUM_ROWS
      .map((row) => {
        const count = hits.filter((hit) => hit.kind === row.kind).length;
        return count ? `${row.label} ${count}` : "";
      })
      .filter(Boolean);
    note.textContent = parts.length
      ? `${parts.join("，")}。格子按 16 分音符对齐到拍子上。`
      : "这一轨没有识别到鼓点。";
  } else {
    note.textContent = `${meta.label}（${meta.en}）识别到 ${(stem.notes || []).length} 个音。横轴是时间，竖轴是音高。`;
  }
  hint.textContent = listenMode === "solo"
    ? `正在只听${meta.label}。`
    : listenMode === "all"
      ? "正在把四条音轨合在一起听。"
      : `正在听原曲，画面是${meta.label}。`;
}

function selectStem(stem) {
  viewStem = stem;
  document.querySelectorAll(".stem-tabs button").forEach((button) => {
    button.classList.toggle("on", button.dataset.stem === stem);
  });
  describeStem();
  renderNoteTable();
  renderDownloads();
  paint(currentTime());
  if (listenMode === "solo") restartIfPlaying();
}

async function loadBuffers(id) {
  audioCtx = audioCtx || new AudioContext();
  const names = ["mix", "vocals", "drums", "bass", "other"];
  await Promise.all(names.map(async (name) => {
    const response = await fetch(`/api/jobs/${id}/audio/${name}`);
    if (!response.ok) throw new Error(name);
    const data = await response.arrayBuffer();
    buffers[name] = await audioCtx.decodeAudioData(data.slice(0));
  }));
}

async function present(data) {
  result = data.result;
  job = data;
  show("result");
  document.querySelector("#song-name").textContent = result.filename || data.filename || "";
  document.querySelector("#bpm-readout").textContent = `${result.bpm} BPM`;
  document.querySelector("#key-readout").textContent = result.key ? result.key.name : "不确定";
  document.querySelector("#time-total").textContent = formatTime(result.duration);
  const pitched = ["vocals", "bass", "other"]
    .filter((name) => (result.stems[name].notes || []).length)
    .sort((a, b) => result.stems[b].notes.length - result.stems[a].notes.length);
  viewStem = pitched[0] || ((result.stems.drums.hits || []).length ? "drums" : "vocals");
  listenMode = "mix";
  document.querySelectorAll(".listen button").forEach((button) => {
    button.classList.toggle("on", button.dataset.listen === "mix");
  });
  try {
    await loadBuffers(data.id);
  } catch {
    showError("结果已经出来了，但播放器没有读到音频。请关掉听音识谱，再从应用程序菜单重新打开。");
    return;
  }
  renderChords();
  selectStem(viewStem);
}

function resetToDrop() {
  stopPlayback();
  pauseOffset = 0;
  buffers = {};
  result = null;
  history.replaceState(null, "", "/");
  followPlayback = true;
  syncTransport();
  show("drop");
}

document.querySelector("#pick-file").addEventListener("click", (event) => {
  event.stopPropagation();
  fileInput.click();
});
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") fileInput.click();
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) upload(fileInput.files[0]);
});
["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => {
  event.preventDefault();
  dropzone.classList.add("hot");
}));
["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => {
  event.preventDefault();
  dropzone.classList.remove("hot");
}));
dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) upload(file);
});
document.querySelector("#demo-button").addEventListener("click", startDemo);
document.querySelector("#play").addEventListener("click", togglePlay);
document.querySelector("#stop").addEventListener("click", () => {
  if (!result) return;
  stopToStart();
});
document.querySelector("#follow").addEventListener("click", () => {
  followPlayback = true;
  syncTransport();
  if (result) paint(currentTime());
});
["#roll-scroll", ".table-scroll"].forEach((selector) => {
  const element = document.querySelector(selector);
  element.addEventListener("scroll", userMovedScroll, { passive: true });
  element.addEventListener("wheel", () => {
    if (playing) pauseFollow();
  }, { passive: true });
  element.addEventListener("pointerdown", (event) => {
    if (!playing) return;
    const box = element.getBoundingClientRect();
    const onVerticalBar = event.clientX >= box.right - 18;
    const onHorizontalBar = event.clientY >= box.bottom - 18;
    if (onVerticalBar || onHorizontalBar) pauseFollow();
  });
});
window.addEventListener("scroll", userMovedScroll, { passive: true });
document.querySelector("#scrub").addEventListener("input", () => {
  if (!result) return;
  seek((Number(document.querySelector("#scrub").value) / 1000) * result.duration);
});
document.querySelector("#roll-scroll").addEventListener("click", (event) => {
  if (!result) return;
  const rect = document.querySelector("#roll-inner").getBoundingClientRect();
  seek((event.clientX - rect.left) / pxPerSec);
});
document.querySelectorAll(".stem-tabs button").forEach((button) => {
  button.addEventListener("click", () => selectStem(button.dataset.stem));
});
document.querySelectorAll(".listen button").forEach((button) => {
  button.addEventListener("click", () => {
    listenMode = button.dataset.listen;
    document.querySelectorAll(".listen button").forEach((item) => item.classList.toggle("on", item === button));
    describeStem();
    restartIfPlaying();
  });
});
document.querySelector("#again").addEventListener("click", resetToDrop);
document.querySelector("#error-back").addEventListener("click", resetToDrop);
window.addEventListener("keydown", (event) => {
  if (!result || event.code !== "Space" || event.repeat) return;
  const tag = event.target && event.target.tagName;
  if (tag === "TEXTAREA" || tag === "SELECT") return;
  if (tag === "INPUT" && event.target.id !== "scrub") return;
  if (event.target && event.target.id === "play") return;
  event.preventDefault();
  togglePlay();
});
window.addEventListener("resize", () => {
  if (!result) return;
  renderChords();
  paint(currentTime());
});

const params = new URLSearchParams(location.search);
if (params.get("job")) poll(params.get("job"));
else show("drop");
