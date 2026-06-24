/*
 * Grove review UI — frontend logic (SPEC.md §6.8, §8, §12).
 *
 * Vanilla JS, no framework, no build step. Talks to the FastAPI review server
 * over the REST API. The whole point of this tool is to turn the teacher's RAW
 * auto-labels (drafts) into TRUSTWORTHY label data — so correctness of the
 * coordinates is the only thing that matters here.
 *
 * ============================ COORDINATE MODEL ============================
 * The server speaks ONLY canonical coords: normalized xyxy, TOP-LEFT origin,
 * each value in [0, 1] (SPEC.md §8). Our in-memory box model also stores
 * canonical coords. The <canvas> draws in device pixels. We therefore convert
 * between the two spaces in EXACTLY ONE place each direction:
 *
 *   canonicalToCanvas(): (x in [0,1]) * imgW_on_canvas + offsetX, likewise y.
 *   canvasToCanonical(): (px - offsetX) / imgW_on_canvas, clamped to [0,1].
 *
 * The image is letterboxed: scaled to FIT the stage preserving aspect ratio,
 * then centered, leaving margins (offsetX/offsetY). `view` holds the scale and
 * offsets. Forgetting the letterbox offset is the classic silent coordinate bug
 * (§12: "boxes come out mirrored, transposed, or off by a constant"), so the
 * offset is applied in both directions and nowhere else.
 * =========================================================================
 */

"use strict";

// ----------------------------------------------------------------------------
// API helpers
// ----------------------------------------------------------------------------
const API = {
  async meta() { return getJSON("/api/meta"); },
  async images() { return getJSON("/api/images"); },
  async image(id) { return getJSON(`/api/images/${encodeURIComponent(id)}`); },
  fileURL(id) { return `/api/images/${encodeURIComponent(id)}/file`; },
  async putBoxes(id, boxes) {
    return postJSON(`/api/images/${encodeURIComponent(id)}/boxes`, { boxes }, "PUT");
  },
  async setStatus(id, status) {
    return postJSON(`/api/images/${encodeURIComponent(id)}/status`, { status }, "POST");
  },
  async export() { return postJSON("/api/export", {}, "POST"); },
};

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`GET ${url} -> ${r.status}`);
  return r.json();
}
async function postJSON(url, body, method) {
  const r = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${method} ${url} -> ${r.status}`);
  return r.json();
}

// ----------------------------------------------------------------------------
// Application state
// ----------------------------------------------------------------------------
const state = {
  classes: [],            // class names from /api/meta
  queue: [],              // [{id, status, box_count}]
  currentIndex: -1,       // index into queue of the loaded image
  current: null,          // {id, width, height, status}
  boxes: [],              // [{label, score, x1,y1,x2,y2}] CANONICAL — source of truth
  selected: -1,           // index into boxes, or -1
  img: null,              // HTMLImageElement of the prepared image
  view: { scale: 1, offsetX: 0, offsetY: 0, imgW: 0, imgH: 0 },
  mode: "select",         // "select" | "newbox"
  // interaction in progress
  drag: null,             // {type, ...} while pointer is down; see pointer handlers
  dirty: false,           // unsaved box edits for the current image
  saveTimer: null,
};

// DOM refs
const els = {};
function $(id) { return document.getElementById(id); }

// Geometry constants (canvas pixels)
const HANDLE = 8;          // half-size of a resize handle hit area
const MIN_BOX_PX = 6;      // ignore drags smaller than this when creating

// ----------------------------------------------------------------------------
// Boot
// ----------------------------------------------------------------------------
window.addEventListener("DOMContentLoaded", init);

async function init() {
  cacheEls();
  wireToolbar();
  wireKeyboard();
  wireCanvas();
  window.addEventListener("resize", () => { layoutCanvas(); render(); });

  try {
    const meta = await API.meta();
    state.classes = meta.classes || [];
    els.projectName.textContent = meta.project_name || "Grove";
    populateClassSelect();
    applyProgress(meta.progress);
  } catch (e) {
    toast("Failed to load /api/meta: " + e.message, "error");
  }

  await refreshQueue();
  if (state.queue.length) {
    // Jump to the first not-yet-reviewed image so resuming a session lands the
    // reviewer where they left off, not back at image 0.
    const start = state.queue.findIndex((q) => q.status === "pending");
    await loadImageAt(start >= 0 ? start : 0);
  } else {
    els.empty.textContent = "No images in this dataset.";
  }
}

function cacheEls() {
  els.app = $("app");
  els.sidebar = $("sidebar");
  els.queue = $("queue");
  els.projectName = $("project-name");
  els.progressText = $("progress-text");
  els.progressFill = $("progress-fill");
  els.countPending = $("count-pending");
  els.countSkipped = $("count-skipped");
  els.currentId = $("current-id");
  els.currentStatus = $("current-status");
  els.classSelect = $("class-select");
  els.canvas = $("canvas");
  els.stage = $("stage");
  els.empty = $("empty-state");
  els.hint = $("hint");
  els.saveState = $("save-state");
  els.helpOverlay = $("help-overlay");
  els.toast = $("toast");
  els.ctx = els.canvas.getContext("2d");
}

// ----------------------------------------------------------------------------
// Class dropdown
// ----------------------------------------------------------------------------
function populateClassSelect() {
  els.classSelect.innerHTML = "";
  state.classes.forEach((name, i) => {
    const opt = document.createElement("option");
    opt.value = name;
    // Show the number-key shortcut alongside the name (1-based).
    opt.textContent = i < 9 ? `${i + 1}. ${name}` : name;
    els.classSelect.appendChild(opt);
  });
}

// ----------------------------------------------------------------------------
// Queue / progress
// ----------------------------------------------------------------------------
async function refreshQueue() {
  try {
    const data = await API.images();
    state.queue = data.images || [];
    applyProgress(data.progress);
    renderQueue();
  } catch (e) {
    toast("Failed to load queue: " + e.message, "error");
  }
}

function applyProgress(p) {
  if (!p) return;
  els.progressText.textContent = `${p.reviewed} / ${p.total} reviewed`;
  els.countPending.textContent = p.pending;
  els.countSkipped.textContent = p.skipped;
  const pct = p.total ? (p.reviewed / p.total) * 100 : 0;
  els.progressFill.style.width = pct.toFixed(1) + "%";
}

function renderQueue() {
  els.queue.innerHTML = "";
  state.queue.forEach((q, i) => {
    const li = document.createElement("li");
    li.className = "queue-item" + (i === state.currentIndex ? " active" : "");
    li.dataset.index = String(i);

    const dot = document.createElement("span");
    dot.className = "dot dot-" + q.status;

    const id = document.createElement("span");
    id.className = "qid";
    id.textContent = q.id;
    id.title = q.id;

    const count = document.createElement("span");
    count.className = "qcount";
    count.textContent = String(q.box_count);
    count.title = q.box_count + " box(es)";

    li.append(dot, id, count);
    li.addEventListener("click", () => loadImageAt(i));
    els.queue.appendChild(li);
  });
  // Keep the active row visible while paging with the keyboard.
  const active = els.queue.querySelector(".queue-item.active");
  if (active) active.scrollIntoView({ block: "nearest" });
}

// Update a single queue row's status/count without a full server round-trip,
// then recompute the progress header locally so the UI feels instant.
function patchQueueRow(index, patch) {
  const row = state.queue[index];
  if (!row) return;
  Object.assign(row, patch);
  renderQueue();
  recomputeProgressFromQueue();
}

function recomputeProgressFromQueue() {
  const p = { total: state.queue.length, reviewed: 0, skipped: 0, pending: 0 };
  for (const q of state.queue) {
    if (q.status === "reviewed") p.reviewed++;
    else if (q.status === "skipped") p.skipped++;
    else p.pending++;
  }
  applyProgress(p);
}

// ----------------------------------------------------------------------------
// Loading an image
// ----------------------------------------------------------------------------
async function loadImageAt(index) {
  if (index < 0 || index >= state.queue.length) return;
  // Persist any pending edits on the outgoing image before we leave it.
  await flushSave();

  state.currentIndex = index;
  const id = state.queue[index].id;

  let meta;
  try {
    meta = await API.image(id);
  } catch (e) {
    toast("Failed to load image " + id + ": " + e.message, "error");
    return;
  }
  state.current = meta;
  state.boxes = (meta.boxes || []).map((b) => ({
    label: b.label || (state.classes[0] || "object"),
    score: b.score ?? null,
    x1: b.x1, y1: b.y1, x2: b.x2, y2: b.y2,
  }));
  state.selected = -1;
  state.dirty = false;
  setMode("select");

  // Load the prepared image bytes. Render once loaded so the canvas sizes
  // itself to the natural image aspect ratio.
  const image = new Image();
  image.onload = () => {
    state.img = image;
    els.empty.classList.add("hidden");
    layoutCanvas();
    render();
  };
  image.onerror = () => {
    state.img = null;
    els.empty.textContent = "Could not load image bytes.";
    els.empty.classList.remove("hidden");
  };
  image.src = API.fileURL(id) + "?t=" + Date.now(); // cache-bust per load

  // Toolbar/header reflect the new image immediately.
  els.currentId.textContent = id;
  els.currentId.title = id;
  setStatusBadge(meta.status);
  renderQueue();
  syncSaveState();
}

function setStatusBadge(status) {
  els.currentStatus.textContent = status;
  els.currentStatus.className = "status-badge status-" + status;
}

// ----------------------------------------------------------------------------
// Canvas layout + the canonical <-> canvas coordinate mapping
// ----------------------------------------------------------------------------
function layoutCanvas() {
  // Match the canvas backing store to its CSS box * devicePixelRatio so lines
  // are crisp on Retina, then compute the letterboxed image placement.
  const rect = els.stage.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  els.canvas.style.width = rect.width + "px";
  els.canvas.style.height = rect.height + "px";
  els.canvas.width = Math.max(1, Math.round(rect.width * dpr));
  els.canvas.height = Math.max(1, Math.round(rect.height * dpr));
  // Draw in CSS pixels; the ctx transform handles dpr so the rest of the code
  // can think purely in CSS pixels.
  els.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const cw = rect.width, ch = rect.height;
  if (!state.img) {
    state.view = { scale: 1, offsetX: 0, offsetY: 0, imgW: cw, imgH: ch };
    return;
  }
  const iw = state.img.naturalWidth, ih = state.img.naturalHeight;
  // Fit preserving aspect ratio; never upscale past a small factor so tiny
  // images don't blow up and blur (cap at 1x to keep boxes pixel-honest).
  const scale = Math.min(cw / iw, ch / ih);
  const imgW = iw * scale, imgH = ih * scale;
  state.view = {
    scale,
    offsetX: (cw - imgW) / 2,   // letterbox margins, centered
    offsetY: (ch - imgH) / 2,
    imgW, imgH,
  };
}

// Canonical (normalized, image space) -> canvas CSS pixels.
function canonXToCanvas(nx) { return state.view.offsetX + nx * state.view.imgW; }
function canonYToCanvas(ny) { return state.view.offsetY + ny * state.view.imgH; }

// Canvas CSS pixels -> canonical normalized, clamped to [0,1] (the image area).
function canvasXToCanon(px) {
  const v = (px - state.view.offsetX) / (state.view.imgW || 1);
  return Math.min(1, Math.max(0, v));
}
function canvasYToCanon(py) {
  const v = (py - state.view.offsetY) / (state.view.imgH || 1);
  return Math.min(1, Math.max(0, v));
}

// Pointer event -> canvas CSS pixel coords (relative to the canvas element).
function eventToCanvas(ev) {
  const rect = els.canvas.getBoundingClientRect();
  return { x: ev.clientX - rect.left, y: ev.clientY - rect.top };
}

// ----------------------------------------------------------------------------
// Rendering
// ----------------------------------------------------------------------------
function render() {
  const ctx = els.ctx;
  const v = state.view;
  // Clear in CSS-pixel space (transform already accounts for dpr).
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);

  if (!state.img) return;
  ctx.drawImage(state.img, v.offsetX, v.offsetY, v.imgW, v.imgH);

  const sel = getComputedStyle(document.documentElement);
  const boxColor = sel.getPropertyValue("--box").trim() || "#4ea1ff";
  const selColor = sel.getPropertyValue("--box-sel").trim() || "#ff5d5d";
  const handleColor = sel.getPropertyValue("--handle").trim() || "#fff";

  state.boxes.forEach((b, i) => {
    const x = canonXToCanvas(b.x1);
    const y = canonYToCanvas(b.y1);
    const w = canonXToCanvas(b.x2) - x;
    const h = canonYToCanvas(b.y2) - y;
    const isSel = i === state.selected;

    ctx.lineWidth = isSel ? 2.5 : 1.8;
    ctx.strokeStyle = isSel ? selColor : boxColor;
    ctx.strokeRect(x, y, w, h);

    // Label chip (label + score if present) anchored at the top-left corner.
    const score = b.score != null ? ` ${(b.score).toFixed(2)}` : "";
    const text = `${b.label}${score}`;
    ctx.font = "12px ui-monospace, Menlo, monospace";
    const tw = ctx.measureText(text).width + 8;
    const ty = y - 16 >= 0 ? y - 16 : y; // flip inside if it would clip the top
    ctx.fillStyle = isSel ? selColor : boxColor;
    ctx.fillRect(x, ty, tw, 16);
    ctx.fillStyle = "#0b1118";
    ctx.fillText(text, x + 4, ty + 12);

    // Resize handles only on the selected box (keeps the view uncluttered).
    if (isSel) {
      ctx.fillStyle = handleColor;
      for (const [hx, hy] of handlePoints(x, y, w, h)) {
        ctx.fillRect(hx - HANDLE / 2, hy - HANDLE / 2, HANDLE, HANDLE);
      }
    }
  });

  // Live preview of the box being drawn in new-box mode.
  if (state.drag && state.drag.type === "create") {
    const d = state.drag;
    const x = Math.min(d.startX, d.curX), y = Math.min(d.startY, d.curY);
    const w = Math.abs(d.curX - d.startX), h = Math.abs(d.curY - d.startY);
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 1.8;
    ctx.strokeStyle = selColor;
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
  }
}

// The 8 handle positions for a box, indexed to match HANDLE_DIRS below.
function handlePoints(x, y, w, h) {
  const mx = x + w / 2, my = y + h / 2;
  return [
    [x, y], [mx, y], [x + w, y],          // nw, n, ne
    [x, my],          [x + w, my],         // w,     e
    [x, y + h], [mx, y + h], [x + w, y + h], // sw, s, se
  ];
}
// Which edges each handle index moves (dx1,dy1,dx2,dy2 flags).
const HANDLE_DIRS = [
  { l: 1, t: 1 }, { t: 1 }, { r: 1, t: 1 },
  { l: 1 },       { r: 1 },
  { l: 1, b: 1 }, { b: 1 }, { r: 1, b: 1 },
];

// ----------------------------------------------------------------------------
// Hit testing (all in canvas CSS pixels)
// ----------------------------------------------------------------------------
function boxRectPx(b) {
  const x = canonXToCanvas(b.x1), y = canonYToCanvas(b.y1);
  return { x, y, w: canonXToCanvas(b.x2) - x, h: canonYToCanvas(b.y2) - y };
}

// If a handle of the selected box is under the pointer, return its index.
function hitHandle(px, py) {
  if (state.selected < 0) return -1;
  const r = boxRectPx(state.boxes[state.selected]);
  const pts = handlePoints(r.x, r.y, r.w, r.h);
  for (let i = 0; i < pts.length; i++) {
    const [hx, hy] = pts[i];
    if (Math.abs(px - hx) <= HANDLE && Math.abs(py - hy) <= HANDLE) return i;
  }
  return -1;
}

// Topmost box whose body contains the point (search reverse = last drawn first).
function hitBox(px, py) {
  for (let i = state.boxes.length - 1; i >= 0; i--) {
    const r = boxRectPx(state.boxes[i]);
    if (px >= r.x && px <= r.x + r.w && py >= r.y && py <= r.y + r.h) return i;
  }
  return -1;
}

// ----------------------------------------------------------------------------
// Pointer interaction: create / select / move / resize
// ----------------------------------------------------------------------------
function wireCanvas() {
  els.canvas.addEventListener("pointerdown", onPointerDown);
  els.canvas.addEventListener("pointermove", onPointerMove);
  els.canvas.addEventListener("pointerup", onPointerUp);
  els.canvas.addEventListener("pointerleave", onPointerUp);
}

function onPointerDown(ev) {
  if (!state.img) return;
  els.canvas.setPointerCapture(ev.pointerId);
  const { x, y } = eventToCanvas(ev);

  // New-box mode: any press starts a create-drag.
  if (state.mode === "newbox") {
    state.drag = { type: "create", startX: x, startY: y, curX: x, curY: y };
    return;
  }

  // 1) Handle of the already-selected box -> resize.
  const hi = hitHandle(x, y);
  if (hi >= 0) {
    state.drag = { type: "resize", handle: hi, box: { ...state.boxes[state.selected] } };
    return;
  }

  // 2) A box body -> select + start move.
  const bi = hitBox(x, y);
  if (bi >= 0) {
    state.selected = bi;
    syncSelectToDropdown();
    state.drag = {
      type: "move",
      box: { ...state.boxes[bi] },
      // Pointer offset within the box, in canonical units, so dragging doesn't
      // snap the box's corner to the cursor.
      grabDX: canvasXToCanon(x) - state.boxes[bi].x1,
      grabDY: canvasYToCanon(y) - state.boxes[bi].y1,
    };
    render();
    return;
  }

  // 3) Empty space in select mode -> begin drawing a new box (convenience: you
  // don't have to press N first; matches typical annotation tools).
  state.selected = -1;
  state.drag = { type: "create", startX: x, startY: y, curX: x, curY: y };
  render();
}

function onPointerMove(ev) {
  if (!state.drag || !state.img) return;
  const { x, y } = eventToCanvas(ev);
  const d = state.drag;

  if (d.type === "create") {
    d.curX = x; d.curY = y;
    render();
    return;
  }

  if (d.type === "move") {
    const b = state.boxes[state.selected];
    const w = d.box.x2 - d.box.x1, h = d.box.y2 - d.box.y1;
    let nx1 = canvasXToCanon(x) - d.grabDX;
    let ny1 = canvasYToCanon(y) - d.grabDY;
    // Clamp so the whole box stays within [0,1] (keeps it on the image).
    nx1 = Math.min(1 - w, Math.max(0, nx1));
    ny1 = Math.min(1 - h, Math.max(0, ny1));
    b.x1 = nx1; b.y1 = ny1; b.x2 = nx1 + w; b.y2 = ny1 + h;
    markDirty();
    render();
    return;
  }

  if (d.type === "resize") {
    const dir = HANDLE_DIRS[d.handle];
    const b = state.boxes[state.selected];
    const cx = canvasXToCanon(x), cy = canvasYToCanon(y);
    // Start from the box as it was at grab time, move only the edges this
    // handle controls, then normalize so x1<x2 / y1<y2 even if dragged across.
    let { x1, y1, x2, y2 } = d.box;
    if (dir.l) x1 = cx;
    if (dir.r) x2 = cx;
    if (dir.t) y1 = cy;
    if (dir.b) y2 = cy;
    b.x1 = Math.min(x1, x2); b.x2 = Math.max(x1, x2);
    b.y1 = Math.min(y1, y2); b.y2 = Math.max(y1, y2);
    markDirty();
    render();
    return;
  }
}

function onPointerUp(ev) {
  if (!state.drag) return;
  const d = state.drag;

  if (d.type === "create") {
    const w = Math.abs(d.curX - d.startX), h = Math.abs(d.curY - d.startY);
    if (w >= MIN_BOX_PX && h >= MIN_BOX_PX) {
      // Convert the pixel drag rect to canonical, choosing the chosen class.
      const x1 = canvasXToCanon(Math.min(d.startX, d.curX));
      const y1 = canvasYToCanon(Math.min(d.startY, d.curY));
      const x2 = canvasXToCanon(Math.max(d.startX, d.curX));
      const y2 = canvasYToCanon(Math.max(d.startY, d.curY));
      const label = els.classSelect.value || state.classes[0] || "object";
      state.boxes.push({ label, score: null, x1, y1, x2, y2 });
      state.selected = state.boxes.length - 1;
      syncSelectToDropdown();
      markDirty();
    }
    setMode("select"); // one box per New-box activation, then back to select
  } else if (d.type === "move" || d.type === "resize") {
    // Drop boxes that collapsed to a sliver during resize (invalid downstream).
    const b = state.boxes[state.selected];
    if (b && (b.x2 - b.x1 < 0.001 || b.y2 - b.y1 < 0.001)) {
      state.boxes.splice(state.selected, 1);
      state.selected = -1;
    }
  }

  state.drag = null;
  scheduleSave();
  render();
}

// ----------------------------------------------------------------------------
// Selection / class helpers
// ----------------------------------------------------------------------------
function syncSelectToDropdown() {
  if (state.selected < 0) return;
  const label = state.boxes[state.selected].label;
  if (state.classes.includes(label)) els.classSelect.value = label;
}

function relabelSelected(label) {
  if (state.selected < 0 || !label) return;
  state.boxes[state.selected].label = label;
  markDirty();
  scheduleSave();
  render();
}

function deleteSelected() {
  if (state.selected < 0) return;
  state.boxes.splice(state.selected, 1);
  state.selected = -1;
  markDirty();
  scheduleSave();
  render();
}

// ----------------------------------------------------------------------------
// Mode
// ----------------------------------------------------------------------------
function setMode(mode) {
  state.mode = mode;
  document.body.classList.toggle("mode-newbox", mode === "newbox");
  $("btn-newbox").classList.toggle("active", mode === "newbox");
  els.hint.textContent = mode === "newbox"
    ? "New-box mode: drag to draw one box"
    : "Drag on empty space to draw a box · click a box to select · drag handles to resize";
}

// ----------------------------------------------------------------------------
// Persistence (debounced PUT of boxes; explicit POST of status)
// ----------------------------------------------------------------------------
function markDirty() { state.dirty = true; syncSaveState(); }

function syncSaveState() {
  if (state.dirty) { els.saveState.textContent = "unsaved"; els.saveState.className = "save-state saving"; }
  else { els.saveState.textContent = "saved"; els.saveState.className = "save-state"; }
}

function scheduleSave() {
  if (state.saveTimer) clearTimeout(state.saveTimer);
  // Debounce so a drag (many edits) results in ONE PUT.
  state.saveTimer = setTimeout(flushSave, 400);
}

async function flushSave() {
  if (state.saveTimer) { clearTimeout(state.saveTimer); state.saveTimer = null; }
  if (!state.dirty || !state.current) return;
  const id = state.current.id;
  const payload = state.boxes.map((b) => ({
    label: b.label, x1: b.x1, y1: b.y1, x2: b.x2, y2: b.y2,
    ...(b.score != null ? { score: b.score } : {}),
  }));
  try {
    await API.putBoxes(id, payload);
    state.dirty = false;
    syncSaveState();
    // Reflect the new box count in the queue row.
    patchQueueRow(state.currentIndex, { box_count: state.boxes.length });
  } catch (e) {
    els.saveState.textContent = "save failed";
    els.saveState.className = "save-state error";
    toast("Save failed: " + e.message, "error");
  }
}

async function markStatus(status) {
  if (!state.current) return;
  await flushSave(); // never lose box edits behind a status change
  const id = state.current.id;
  try {
    await API.setStatus(id, status);
    state.current.status = status;
    setStatusBadge(status);
    patchQueueRow(state.currentIndex, { status });
  } catch (e) {
    toast("Status update failed: " + e.message, "error");
  }
}

// ----------------------------------------------------------------------------
// Navigation
// ----------------------------------------------------------------------------
function next() { if (state.currentIndex < state.queue.length - 1) loadImageAt(state.currentIndex + 1); }
function prev() { if (state.currentIndex > 0) loadImageAt(state.currentIndex - 1); }

// ----------------------------------------------------------------------------
// Export
// ----------------------------------------------------------------------------
async function doExport() {
  await flushSave();
  toast("Exporting dataset…", "ok");
  try {
    const res = await API.export();
    const summary = res && res.summary ? JSON.stringify(res.summary, null, 2) : "done";
    toast("Export complete:\n" + summary, "ok", 6000);
  } catch (e) {
    toast("Export failed: " + e.message, "error", 6000);
  }
}

// ----------------------------------------------------------------------------
// Toolbar + keyboard wiring
// ----------------------------------------------------------------------------
function wireToolbar() {
  $("btn-prev").addEventListener("click", prev);
  $("btn-next").addEventListener("click", next);
  $("btn-newbox").addEventListener("click", () => setMode(state.mode === "newbox" ? "select" : "newbox"));
  $("btn-delete").addEventListener("click", deleteSelected);
  $("btn-reviewed").addEventListener("click", async () => { await markStatus("reviewed"); next(); });
  $("btn-skipped").addEventListener("click", () => markStatus("skipped"));
  $("btn-export").addEventListener("click", doExport);
  $("btn-help").addEventListener("click", toggleHelp);
  $("help-close").addEventListener("click", toggleHelp);
  els.classSelect.addEventListener("change", () => relabelSelected(els.classSelect.value));
}

function wireKeyboard() {
  document.addEventListener("keydown", (ev) => {
    // Don't hijack typing in form controls (the class <select>, etc.).
    const tag = (ev.target && ev.target.tagName) || "";
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") {
      if (ev.key === "Escape") ev.target.blur();
      return;
    }

    switch (ev.key) {
      case "ArrowRight": case "d": case "D": ev.preventDefault(); next(); break;
      case "ArrowLeft":  case "a": case "A": ev.preventDefault(); prev(); break;
      case "n": case "N": ev.preventDefault(); setMode(state.mode === "newbox" ? "select" : "newbox"); break;
      case "Delete": case "Backspace": ev.preventDefault(); deleteSelected(); break;
      case "r": case "R": ev.preventDefault(); (async () => { await markStatus("reviewed"); next(); })(); break;
      case "s": case "S": ev.preventDefault(); markStatus("skipped"); break;
      case "Escape":
        ev.preventDefault();
        if (state.drag && state.drag.type === "create") { state.drag = null; }
        state.selected = -1;
        setMode("select");
        render();
        break;
      case "?": ev.preventDefault(); toggleHelp(); break;
      default:
        // Number keys 1..9 -> set the selected box's class.
        if (/^[1-9]$/.test(ev.key)) {
          const idx = parseInt(ev.key, 10) - 1;
          if (idx < state.classes.length) {
            ev.preventDefault();
            els.classSelect.value = state.classes[idx];
            relabelSelected(state.classes[idx]);
          }
        }
    }
  });
}

function toggleHelp() { els.helpOverlay.classList.toggle("hidden"); }

// ----------------------------------------------------------------------------
// Toast
// ----------------------------------------------------------------------------
let toastTimer = null;
function toast(msg, kind, ms) {
  els.toast.textContent = msg;
  els.toast.className = "toast " + (kind || "");
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => els.toast.classList.add("hidden"), ms || 3000);
}

// Best-effort flush if the user closes the tab with unsaved edits. We can't use
// navigator.sendBeacon here because it only issues POST and the boxes route is
// PUT; instead we fire a keepalive fetch, which survives unload in modern
// browsers. The primary safety net is still flushSave() on every navigation.
window.addEventListener("beforeunload", () => {
  if (state.dirty && state.current) {
    const id = state.current.id;
    const payload = state.boxes.map((b) => ({
      label: b.label, x1: b.x1, y1: b.y1, x2: b.x2, y2: b.y2,
      ...(b.score != null ? { score: b.score } : {}),
    }));
    fetch(`/api/images/${encodeURIComponent(id)}/boxes`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ boxes: payload }),
      keepalive: true,
    }).catch(() => {});
  }
});
