"use strict";
const el = (id) => document.getElementById(id);
const put = (id, text) => {
  el(id).textContent = text;
};
const make = (tag, text) => {
  const item = document.createElement(tag);
  if (text != null) item.textContent = text;
  return item;
};
let mode = "controlled",
  run = "verified",
  steps = [],
  idx = 1,
  frame = null,
  token = 0,
  clock = null,
  poll = null,
  follow = false;
let cells,
  odds,
  gridBox,
  selected = null;
const base = () => "/runs/" + run + "/" + mode + "/";
async function json(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) throw Error("HTTP " + res.status);
  return res.json();
}
function stop() {
  clearInterval(clock);
  clock = null;
  put("play", "播放");
}
function xyCell(xy) {
  return [
    Math.floor((xy[1] - frame.origin[1]) / frame.resolution),
    Math.floor((xy[0] - frame.origin[0]) / frame.resolution),
  ];
}
function paintMap() {
  const cv = el("map"),
    ctx = cv.getContext("2d"),
    [rows, cols] = frame.shape,
    w = cv.width,
    h = cv.height;
  const size = Math.min((w - 38) / cols, (h - 24) / rows),
    ox = (w - cols * size) / 2,
    oy = (h - rows * size) / 2;
  gridBox = { size, ox, oy };
  ctx.fillStyle = "#eaf0f1";
  ctx.fillRect(0, 0, w, h);
  const tiny = document.createElement("canvas");
  tiny.width = cols;
  tiny.height = rows;
  const gc = tiny.getContext("2d"),
    im = gc.createImageData(cols, rows);
  for (let i = 0; i < cells.length; i++) {
    const c =
      cells[i] === 255
        ? [185, 201, 206]
        : cells[i] === 0
          ? [252, 252, 248]
          : [21, 57, 72];
    im.data.set([...c, 255], i * 4);
  }
  for (const [r, c] of frame.visits)
    im.data.set([237, 160, 80, 255], 4 * (r * cols + c));
  gc.putImageData(im, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(tiny, ox, oy, cols * size, rows * size);
  if (rows < 36) {
    ctx.strokeStyle = "#849ca54d";
    ctx.lineWidth = 0.8;
    for (let i = 0; i <= cols; i++) {
      ctx.beginPath();
      ctx.moveTo(ox + i * size, oy);
      ctx.lineTo(ox + i * size, oy + rows * size);
      ctx.stroke();
    }
    for (let i = 0; i <= rows; i++) {
      ctx.beginPath();
      ctx.moveTo(ox, oy + i * size);
      ctx.lineTo(ox + cols * size, oy + i * size);
      ctx.stroke();
    }
  }
  const point = (xy) => [
    ox + ((xy[0] - frame.origin[0]) * size) / frame.resolution,
    oy + ((xy[1] - frame.origin[1]) * size) / frame.resolution,
  ];
  if (frame.ray) {
    ctx.beginPath();
    ctx.moveTo(...point(frame.ray[0]));
    ctx.lineTo(...point(frame.ray[1]));
    ctx.strokeStyle = "#ec8036";
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.setLineDash([]);
  }
  const [x, y] = point(frame.xy);
  ctx.beginPath();
  ctx.arc(x, y, 5, 0, 2 * Math.PI);
  ctx.fillStyle = "#ed573d";
  ctx.fill();
  ctx.strokeStyle = "white";
  ctx.lineWidth = 1.5;
  ctx.stroke();
  if (selected) {
    const [r, c] = selected;
    ctx.strokeStyle = "#a42d66";
    ctx.lineWidth = 2;
    ctx.strokeRect(ox + c * size, oy + r * size, size, size);
  }
  ctx.fillStyle = "#315461";
  ctx.font = "14px sans-serif";
  ctx.fillText("+X →", 8, 21);
  ctx.fillText("+Z ↓", 8, 41);
}
function inspect() {
  if (!selected) {
    put("inspect", "点击任意栅格查看行列、占用证据和访问次数。");
    return;
  }
  const [r, c] = selected,
    i = r * frame.shape[1] + c,
    v = frame.visits.find((t) => t[0] === r && t[1] === c);
  put(
    "inspect",
    "行 " +
      r +
      " · 列 " +
      c +
      "　" +
      { 255: "未知", 0: "空闲", 100: "占用" }[cells[i]] +
      "　log-odds " +
      (odds.getInt16(i * 2, true) / 100).toFixed(2) +
      "　访问 " +
      (v ? v[2] : 0) +
      " 次",
  );
}
el("map").onclick = (e) => {
  if (!frame) return;
  const cv = el("map"),
    rect = cv.getBoundingClientRect(),
    { ox, oy, size } = gridBox;
  const c = Math.floor(
      (((e.clientX - rect.left) * cv.width) / rect.width - ox) / size,
    ),
    r = Math.floor(
      (((e.clientY - rect.top) * cv.height) / rect.height - oy) / size,
    );
  if (r >= 0 && c >= 0 && r < frame.shape[0] && c < frame.shape[1]) {
    selected = [r, c];
    paintMap();
    inspect();
  }
};
function paintGraph() {
  const cv = el("graph"),
    g = cv.getContext("2d"),
    w = cv.width,
    h = cv.height;
  g.fillStyle = "#eef4f4";
  g.fillRect(0, 0, w, h);
  const places = frame.places,
    p0 = places[0],
    byId = {};
  const xs = places.map((p) => p.x),
    ys = places.map((p) => p.y);
  const span = Math.max(
    1.2,
    Math.max(...xs) - Math.min(...xs),
    Math.max(...ys) - Math.min(...ys),
  );
  const cx = (Math.max(...xs) + Math.min(...xs)) / 2,
    cy = (Math.max(...ys) + Math.min(...ys)) / 2;
  for (const p of places)
    byId[p.place_id] = [
      w / 2 + ((p.x - cx) * 235) / span,
      h / 2 + ((p.y - cy) * 235) / span,
    ];
  g.lineWidth = 4;
  g.textAlign = "center";
  g.font = "18px sans-serif";
  for (const e of frame.edges) {
    const [x, y] = byId[e.source_id],
      [u, v] = byId[e.target_id],
      ang = Math.atan2(v - y, u - x),
      dx = Math.cos(ang),
      dy = Math.sin(ang),
      tx = u - 27 * dx,
      ty = v - 27 * dy;
    g.strokeStyle = g.fillStyle = {
      BLOCKED: "#b94e42",
      PLANNED: "#b67d26",
      TRAVERSED: "#107768",
    }[e.status];
    g.setLineDash(e.status === "TRAVERSED" ? [] : [8, 5]);
    g.beginPath();
    g.moveTo(x + 25 * dx, y + 25 * dy);
    g.lineTo(tx, ty);
    g.stroke();
    g.setLineDash([]);
    g.beginPath();
    g.moveTo(tx, ty);
    g.lineTo(tx - 13 * dx + 6 * dy, ty - 13 * dy - 6 * dx);
    g.lineTo(tx - 13 * dx - 6 * dy, ty - 13 * dy + 6 * dx);
    g.fill();
    g.fillText(e.status, (x + u) / 2 + dy * 29, (y + v) / 2 - dx * 29);
  }
  for (const p of places) {
    const [x, y] = byId[p.place_id];
    g.fillStyle = "#174b59";
    g.beginPath();
    g.arc(x, y, 24, 0, 2 * Math.PI);
    g.fill();
    g.fillStyle = "white";
    g.fillText(p.place_id, x, y + 6);
    g.fillStyle = "#41616a";
    g.font = "15px sans-serif";
    g.fillText("访问 " + p.visit_count, x, y + 45);
    g.font = "18px sans-serif";
  }
  const rec = el("records");
  rec.replaceChildren();
  for (const p of places)
    rec.append(
      make(
        "div",
        p.place_id +
          " · 观测 " +
          (p.observation_ids.join("、") || "暂无") +
          Object.entries(frame.frontiers)
            .filter((t) => t[1] === p.place_id)
            .map((t) => " · 引用 " + t[0])
            .join(""),
      ),
    );
  for (const e of frame.edges)
    rec.append(
      make(
        "div",
        e.source_id +
          " → " +
          e.target_id +
          " · " +
          e.path_cost.toFixed(2) +
          " m · 成功 " +
          e.traversal_count +
          " · 累计失败 " +
          e.failure_count,
      ),
    );
}
function sensors() {
  const wrap = el("inputs");
  wrap.replaceChildren();
  if (!frame.rgb) {
    wrap.append(
      make(
        "p",
        frame.ray
          ? "模拟单像素深度 1.00 m，已知相机朝向 −Z。橙色虚线为观测射线。"
          : frame.rejection
            ? "模块返回：" + frame.rejection
            : frame.detail,
      ),
    );
    return;
  }
  const row = make("div");
  row.className = "imgs";
  for (const [file, label] of [
    [frame.rgb, "Habitat RGB"],
    [frame.depth, "深度 · 黑近白远，0–8 m"],
  ]) {
    const fig = make("figure"),
      img = make("img");
    img.src = base() + file;
    img.alt = label;
    fig.append(img, make("figcaption", label));
    row.append(fig);
  }
  wrap.append(
    row,
    make(
      "p",
      "中心像素深度 " + frame.center_m + " m · 实际动作 " + frame.action,
    ),
  );
  if (frame.witness) {
    const v = frame.witness;
    wrap.append(
      make(
        "p",
        "中心射线命中第 " +
          v.cell[0] +
          " 行、第 " +
          v.cell[1] +
          " 列，log-odds 从 " +
          v.before.toFixed(2) +
          " 变为 " +
          v.after.toFixed(2) +
          "。地图中橙色虚线标出这条射线。",
      ),
    );
  }
  const s = frame.semantic;
  if (!s) return;
  const box = make("div");
  box.className = "sem";
  box.append(
    make("b", "历史感知 · " + s.goal),
    make(
      "p",
      s.model +
        " · " +
        s.id +
        " · 目标候选 " +
        s.entities.length +
        " 个 · 环境物体 " +
        s.context.length +
        " 个",
    ),
  );
  const archive = make("div");
  archive.className = "frame";
  const img = make("img");
  img.src = base() + "archive.jpg";
  img.alt = "历史感知输入和目标候选框";
  archive.append(img);
  for (const e of s.entities) {
    const [x, y, u, v] = e.bbox,
      mark = make("span");
    mark.className = "box";
    Object.assign(mark.style, {
      left: x / 10 + "%",
      top: y / 10 + "%",
      width: (u - x) / 10 + "%",
      height: (v - y) / 10 + "%",
    });
    mark.title = e.local_id + " " + e.category;
    archive.append(mark);
  }
  const det = make("details"),
    summary = make("summary", "候选框与输入核对");
  det.append(
    summary,
    archive,
    make(
      "p",
      "复渲染像素平均绝对差 " +
        s.pixel_mae.toFixed(3) +
        " / 255 。原观测挂接在 P01。",
    ),
    make("p", "源图 SHA-256 " + s.sha),
  );
  det.style.overflowWrap = "anywhere";
  box.append(det);
  wrap.append(box);
}
async function show(i) {
  if (!steps.length) return;
  const ticket = ++token;
  idx = Math.min(steps.length - 1, Math.max(0, i));
  const url = base() + idx + ".json";
  const f = await json(url);
  if (ticket !== token) return;
  frame = f;
  cells = Uint8Array.from(atob(f.cells), (c) => c.charCodeAt(0));
  odds = new DataView(
    Uint8Array.from(atob(f.odds), (c) => c.charCodeAt(0)).buffer,
  );
  el("slide").value = idx;
  el("list").value = idx;
  el("prev").disabled = idx === 0;
  el("next").disabled = idx === steps.length - 1;
  put("progress", "步骤 " + (idx + 1) + " / " + steps.length);
  put("title", f.title);
  put("detail", f.detail);
  put("scale", f.resolution + " m / 格 · 版本 " + f.version);
  put("graphcount", f.places.length + " 个节点 · " + f.edges.length + " 条边");
  el("metrics").replaceChildren();
  for (const [k, label] of [
    ["free", "空闲"],
    ["occupied", "占用"],
    ["unknown", "未知"],
    ["visits", "进入次数"],
  ]) {
    const a = make("span");
    a.append(make("b", f.counts[k]), " " + label);
    el("metrics").append(a);
  }
  const checks = el("checks");
  checks.replaceChildren();
  for (const c of f.checks) {
    const line = make("div", (c.passed ? "✓　" : "×　") + c.label);
    line.className = c.passed ? "check" : "bad";
    checks.append(line);
  }
  put(
    "change",
    "相对上一步，" +
      f.changed +
      " 格改变占用分类，新增 " +
      f.entered +
      " 次底盘进入记录。",
  );
  el("raw").href = url;
  paintMap();
  inspect();
  paintGraph();
  sensors();
}
async function loadSteps(where, first) {
  const target = base(),
    items = await json(target + "steps.json");
  if (target !== base()) return;
  steps = items;
  const list = el("list");
  list.replaceChildren();
  items.forEach((v, i) => {
    const o = make("option", i + 1 + ". " + v.title);
    o.value = i;
    list.append(o);
  });
  el("slide").max = items.length - 1;
  await show(first ? 0 : where);
}
function manual() {
  follow = false;
  stop();
}
async function tab(m) {
  manual();
  token++;
  mode = m;
  selected = null;
  el("controlled").classList.toggle("active", m === "controlled");
  el("mp3d").classList.toggle("active", m === "mp3d");
  put(
    "source",
    m === "controlled"
      ? "模拟输入 · M1 核心规则"
      : "Habitat-Sim · MP3D / 2azQ1b91cZZ",
  );
  try {
    await loadSteps(0, true);
  } catch (e) {
    put("title", "等待本组首个快照");
    put("detail", e.message);
  }
}
el("controlled").onclick = () => tab("controlled");
el("mp3d").onclick = () => tab("mp3d");
el("prev").onclick = () => {
  manual();
  show(idx - 1);
};
el("next").onclick = () => {
  manual();
  show(idx + 1);
};
el("slide").oninput = (e) => {
  manual();
  show(Number(e.target.value));
};
el("list").onchange = (e) => {
  manual();
  show(Number(e.target.value));
};
el("play").onclick = () => {
  if (clock) {
    stop();
    return;
  }
  follow = false;
  if (idx === steps.length - 1) show(0);
  put("play", "暂停");
  clock = setInterval(() => {
    if (idx < steps.length - 1) show(idx + 1);
    else stop();
  }, 1250);
};
async function track() {
  const st = await json("/api/status");
  el("run").disabled = st.status === "running";
  put(
    "status",
    st.status === "running"
      ? "服务器计算中"
      : st.status === "failed"
        ? "运行失败 · " + st.error
        : "已连接 · 可回放",
  );
  if (st.status === "running" || follow) {
    if (follow) {
      run = st.id;
      let target = mode;
      try {
        await json("/runs/" + run + "/mp3d/steps.json");
        target = "mp3d";
      } catch {}
      if (target !== mode) {
        await tab(target);
        follow = true;
      }
      try {
        await loadSteps(1e6, false);
      } catch {}
    }
    if (st.status !== "running") follow = false;
  }
  if (st.status === "running" && !poll)
    poll = setInterval(
      () => track().catch((e) => put("status", e.message)),
      600,
    );
  if (st.status !== "running" && poll) {
    clearInterval(poll);
    poll = null;
  }
}
el("run").onclick = async () => {
  try {
    stop();
    el("run").disabled = true;
    const st = await json("/api/run", {
      method: "POST",
      headers: { "X-M1-Lab": "run" },
    });
    run = st.id;
    await tab("controlled");
    follow = true;
    await track();
  } catch (e) {
    put("status", e.message);
    el("run").disabled = false;
  }
};
(async () => {
  const s = await json("/api/status");
  run = s.status === "ready" ? s.id : "verified";
  await tab("controlled");
  await track();
})().catch((e) => put("status", "连接异常 · " + e.message));
