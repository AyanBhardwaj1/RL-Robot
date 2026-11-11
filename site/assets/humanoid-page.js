// Humanoid push lab page: click to shove, animate capture-point recovery, live push envelopes.
(function () {
  "use strict";
  const H = window.Humanoid;
  const $ = (s) => document.querySelector(s);
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const n0 = new Intl.NumberFormat("en", { maximumFractionDigits: 0 });
  const n2 = new Intl.NumberFormat("en", { maximumFractionDigits: 2 });
  const pct = new Intl.NumberFormat("en", { style: "percent", maximumFractionDigits: 1 });
  const canvas = $("#hum-canvas");
  const ctx = canvas.getContext("2d");
  let SCALE = 260; // px per metre; zoomed in on narrow screens so the robot stays legible
  const fitScale = () => { SCALE = canvas.clientWidth && canvas.clientWidth < 600 ? 380 : 260; };
  fitScale();
  window.addEventListener("resize", () => { fitScale(); if (last.result) play(last.result); });
  const COLORS = { ground: "#17161b", grid: "#221f27", foot: "#5d5866", footNew: "#8a8494", com: "#f3eee8", cp: "#f47b25", push: "#f99a55", base: "#3a3641" };
  let anim = null;
  let shownCom = null; // screen position of the drawn centre of mass, for click-to-shove
  let last = { dir: -90, result: null };

  function body() { return document.querySelector('input[name="body"]:checked').value; }
  function cond() {
    return { payload_kg: +$("#payload").value, friction: +$("#friction").value, latency_s: +$("#latency").value / 1000,
      actuator_strength: +$("#strength").value, floor_step_m: +$("#step").value };
  }

  // ---------- drawing (robot faces up; its left is screen left) ----------
  function toScreen(x, y, origin) {
    return [canvas.width / 2 + origin.sx - y * SCALE, canvas.height * 0.62 + origin.sy - x * SCALE];
  }

  function drawFrame(res, k, origin) {
    const W = canvas.width, Hh = canvas.height;
    ctx.fillStyle = COLORS.ground; ctx.fillRect(0, 0, W, Hh);
    ctx.fillStyle = COLORS.grid;
    for (let gx = -10; gx <= 10; gx++) { const [sx] = toScreen(0, gx * 0.25, origin); ctx.fillRect(sx, 0, 1, Hh); }
    for (let gy = -10; gy <= 10; gy++) { const [, sy] = toScreen(gy * 0.25, 0, origin); ctx.fillRect(0, sy, W, 1); }
    k = Math.max(0, k);
    const t = res.com.length ? res.com[Math.min(k, res.com.length - 1)][0] : 0;
    const wheeled = body() === "wheeled";
    if (wheeled) {
      const b = res.footsteps.length ? res.footsteps[Math.max(0, Math.min(k, res.footsteps.length - 1))] : [0, 0, 0];
      const [cx, cy] = toScreen(b[1], b[2], origin);
      ctx.fillStyle = COLORS.base;
      ctx.fillRect(cx - H.WHEELED.half_width * SCALE, cy - H.WHEELED.half_length * SCALE, 2 * H.WHEELED.half_width * SCALE, 2 * H.WHEELED.half_length * SCALE);
    } else {
      // starting stance: two feet
      const fl = H.BIPED.foot_half_length * SCALE, fw = H.BIPED.foot_half_width * SCALE;
      for (const side of [1, -1]) {
        const [fx, fy] = toScreen(0, side * H.BIPED.stance_half_width, origin);
        ctx.fillStyle = COLORS.foot; ctx.fillRect(fx - fw, fy - fl, 2 * fw, 2 * fl);
      }
      res.footsteps.filter((f) => f[0] <= t + 1e-9).forEach((f, i) => {
        const [fx, fy] = toScreen(f[1], f[2], origin);
        ctx.fillStyle = COLORS.footNew; ctx.fillRect(fx - fw, fy - fl, 2 * fw, 2 * fl);
        ctx.fillStyle = COLORS.ground; ctx.font = "800 18px Schibsted Grotesk, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(String(i + 1), fx, fy);
      });
    }
    // centre-of-mass trail
    ctx.strokeStyle = COLORS.com; ctx.lineWidth = 3; ctx.beginPath();
    res.com.slice(0, k + 1).forEach((p, i) => { const [sx, sy] = toScreen(p[1], p[2], origin); if (i) ctx.lineTo(sx, sy); else ctx.moveTo(sx, sy); });
    ctx.stroke();
    if (res.com.length) {
      const c = res.com[Math.min(k, res.com.length - 1)];
      const [sx, sy] = toScreen(c[1], c[2], origin);
      shownCom = [sx, sy];
      ctx.fillStyle = COLORS.com; ctx.beginPath(); ctx.arc(sx, sy, 14, 0, Math.PI * 2); ctx.fill();
      const cp = res.capture[Math.min(k, res.capture.length - 1)];
      const [px, py] = toScreen(cp[1], cp[2], origin);
      ctx.fillStyle = COLORS.cp; ctx.beginPath(); ctx.arc(px, py, 9, 0, Math.PI * 2); ctx.fill();
    } else {
      const [sx, sy] = toScreen(0, 0, origin);
      shownCom = [sx, sy];
      ctx.fillStyle = COLORS.com; ctx.beginPath(); ctx.arc(sx, sy, 14, 0, Math.PI * 2); ctx.fill();
    }
    // the shove: an arrow arriving from the side it came from, for the first 0.4 s
    if (t < 0.4 && last.result) {
      const a = last.dir * Math.PI / 180;
      const [ex, ey] = toScreen(0, 0, origin);
      const len = 70 + Math.min(1, +$("#imp").value / 200) * 110;
      const sx = ex - (-Math.sin(a)) * (len + 30), sy = ey - (-Math.cos(a)) * (len + 30);
      const tx = ex - (-Math.sin(a)) * 30, ty = ey - (-Math.cos(a)) * 30;
      ctx.strokeStyle = COLORS.push; ctx.lineWidth = 8; ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(tx, ty); ctx.stroke();
      const hd = Math.atan2(ty - sy, tx - sx);
      ctx.fillStyle = COLORS.push; ctx.beginPath(); ctx.moveTo(tx + Math.cos(hd) * 6, ty + Math.sin(hd) * 6);
      ctx.lineTo(tx - Math.cos(hd - 0.5) * 26, ty - Math.sin(hd - 0.5) * 26); ctx.lineTo(tx - Math.cos(hd + 0.5) * 26, ty - Math.sin(hd + 0.5) * 26); ctx.fill();
    }
    ctx.fillStyle = COLORS.com; ctx.font = "600 20px Schibsted Grotesk, sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "top";
    ctx.fillText(`t ${t.toFixed(2)} s`, 18, 16);
  }

  function originFor(res) {
    // keep the action on screen: centre between start and where the CoM ends up
    if (!res || !res.com.length) return { sx: 0, sy: 0 };
    const end = res.com[res.com.length - 1];
    const mx = Math.max(-0.9, Math.min(0.9, end[1] / 2)), my = Math.max(-1.2, Math.min(1.2, end[2] / 2));
    return { sx: my * SCALE, sy: mx * SCALE };
  }

  function followOrigin(res, k) {
    // camera follows the centre of mass so a fall or a long recovery stays in view
    if (!res.com.length) return { sx: 0, sy: 0 };
    const c = res.com[Math.max(0, Math.min(k, res.com.length - 1))];
    return { sx: c[2] * SCALE * 0.85, sy: c[1] * SCALE * 0.85 };
  }

  function play(res) {
    if (anim) cancelAnimationFrame(anim);
    const origin = originFor(res);
    if (reduced || !res.com.length) { drawFrame(res, res.com.length - 1, res.com.length ? followOrigin(res, res.com.length - 1) : origin); return; }
    const start = performance.now();
    const dt = res.com.length > 1 ? res.com[1][0] - res.com[0][0] : 0.02;
    const tick = (now) => {
      const k = Math.max(0, Math.floor((now - start) / 1000 / dt)); // rAF time can precede start
      drawFrame(res, k, followOrigin(res, k));
      if (k < res.com.length - 1) anim = requestAnimationFrame(tick);
    };
    anim = requestAnimationFrame(tick);
  }

  function describe(r, dir, imp) {
    const gap = (d) => Math.abs(((dir - d + 540) % 360) - 180);
    const side = Object.entries(H.DIRECTIONS).reduce((best, e) => (gap(e[1]) < gap(best[1]) ? e : best))[0];
    if (r.recovered) return [r.steps ? `Saved in ${r.steps} step${r.steps > 1 ? "s" : ""}` : "Held", `${n0.format(imp)} N·s from the ${side}: ${r.steps ? "it stepped where the capture point was heading" : "ankles alone kept the capture point inside the support"}${body() === "wheeled" ? " (the base drove under it)" : ""}.`];
    const reason = r.reason.charAt(0).toUpperCase() + r.reason.slice(1);
    return [r.reason.startsWith("blocked") ? "Blocked" : "Fell", `${n0.format(imp)} N·s from the ${side}. ${reason}.`];
  }

  function shove(dir) {
    const imp = +$("#imp").value;
    last.dir = dir;
    const r = H.recover(body(), imp, dir, cond(), true);
    last.result = r;
    const [word, why] = describe(r, dir, imp);
    $("#h-verdict").textContent = word;
    $("#h-why").textContent = why;
    $(".hum-result").classList.toggle("fail", !r.recovered);
    play(r);
  }

  // ---------- live envelopes ----------
  let envTimer = null;
  function polar(biped, wheel) {
    const svg = $("#h-polar");
    const order = ["front", "front-right", "right", "rear-right", "rear", "rear-left", "left", "front-left"];
    const max = Math.max(60, ...Object.values(biped), ...Object.values(wheel));
    const R = 88;
    const pt = (name, v) => { const a = H.DIRECTIONS[name] * Math.PI / 180; const r = R * Math.min(1, v / max); return `${(Math.sin(a) * r).toFixed(1)},${(Math.cos(a) * r).toFixed(1)}`; };
    let out = "";
    for (const f of [0.5, 1]) out += `<circle class="ring" r="${R * f}"></circle>`;
    for (const name of order) {
      const a = H.DIRECTIONS[name] * Math.PI / 180;
      if (["front", "rear", "left", "right"].includes(name)) out += `<text class="lbl" x="${(Math.sin(a) * (R + 12)).toFixed(1)}" y="${(Math.cos(a) * (R + 12) + 3).toFixed(1)}" text-anchor="middle">${name}</text>`;
    }
    out += `<text class="lbl" x="${(R * 0.5 * 0.71 + 3).toFixed(1)}" y="${(-R * 0.5 * 0.71).toFixed(1)}">${n0.format(max / 2)}</text><text class="lbl" x="${(R * 0.71 + 3).toFixed(1)}" y="${(-R * 0.71).toFixed(1)}">${n0.format(max)}</text>`;
    out += `<polygon class="wheel" points="${order.map((n) => pt(n, wheel[n])).join(" ")}"></polygon>`;
    out += `<polygon class="biped" points="${order.map((n) => pt(n, biped[n])).join(" ")}"></polygon>`;
    svg.innerHTML = out;
    const dl = $("#h-env"); dl.replaceChildren();
    for (const [k, label] of [["front", "From the front"], ["rear", "From the rear"], ["left", "From the left"], ["right", "From the right"]]) {
      const dt = document.createElement("dt"); dt.textContent = label;
      const dd = document.createElement("dd"); dd.textContent = `${n0.format(biped[k])} / ${n0.format(wheel[k])} N·s`;
      dl.append(dt, dd);
    }
  }
  // ---------- capturability map ----------
  const CAP_N = 37, CAP_V = 1.6;
  const CAP_COLORS = { "-1": "#0b0a0d", 0: "#3a3641", 1: "#7b4a2c", 2: "#b55e22", 3: "#f99a55" };
  function drawCapMap() {
    const cv = $("#cap-canvas");
    if (!cv) return;
    const g = H.capturabilityMap(body(), cond(), CAP_N, CAP_V);
    const c2 = cv.getContext("2d");
    const cell = cv.width / CAP_N;
    const counts = { "-1": 0, 0: 0, 1: 0, 2: 0, 3: 0 };
    g.forEach((row, i) => row.forEach((v, j) => {
      counts[v] = (counts[v] || 0) + 1;
      c2.fillStyle = CAP_COLORS[v] || CAP_COLORS[3];
      c2.fillRect(Math.floor(j * cell), Math.floor(i * cell), Math.ceil(cell), Math.ceil(cell));
    }));
    // axes through the centre (standing still) and speed rings every 0.5 m/s
    c2.strokeStyle = "rgba(243,238,232,.35)"; c2.lineWidth = 1;
    c2.beginPath(); c2.moveTo(cv.width / 2, 0); c2.lineTo(cv.width / 2, cv.height); c2.moveTo(0, cv.height / 2); c2.lineTo(cv.width, cv.height / 2); c2.stroke();
    for (const v of [0.5, 1.0, 1.5]) { c2.beginPath(); c2.arc(cv.width / 2, cv.height / 2, v / CAP_V * cv.width / 2, 0, Math.PI * 2); c2.stroke(); }
    c2.fillStyle = "#f3eee8"; c2.font = "600 15px Schibsted Grotesk, sans-serif"; c2.textAlign = "center";
    c2.fillText("forward", cv.width / 2, 18); c2.fillText("backward", cv.width / 2, cv.height - 8);
    c2.textAlign = "left"; c2.fillText("left", 8, cv.height / 2 - 8); c2.textAlign = "right"; c2.fillText("right", cv.width - 8, cv.height / 2 - 8);
    const total = CAP_N * CAP_N, ok = total - counts["-1"];
    $("#cap-stats").textContent = `${pct.format(ok / total)} of shoves up to 1.6 m/s are recoverable: ${counts[0]} with ankles only, ${counts[1]} with 1 step, ${counts[2]} with 2, ${counts[3]} with 3.`;
  }

  function refreshEnvelope() {
    clearTimeout(envTimer);
    envTimer = setTimeout(() => { polar(H.envelope("biped", cond(), 9), H.envelope("wheeled", cond(), 9)); drawCapMap(); }, 60);
  }

  // ---------- jobs table (from the Python run) ----------
  const WHY = {
    "line-side kitting": "Sealed concrete, totes up to 16 kg, people nearby.",
    "tote handling": "Spills down to μ 0.3, totes up to 25 kg.",
    "trailer unloading": "Trailer lips and debris up to 6 cm; ramps of ±5°.",
    "home care": "Thresholds and stairs up to 18 cm on half the routes.",
    "teleoperated data collection": "40–250 ms of network latency on every command.",
  };
  fetch("/assets/data/humanoid.json").then((r) => r.json()).then((d) => {
    const tb = $("#jobs-table tbody");
    Object.entries(d.jobs).forEach(([job, row]) => {
      const tr = document.createElement("tr");
      const cell = (txt, cls) => { const td = document.createElement("td"); td.textContent = txt; if (cls) td.className = cls; return td; };
      const b = row.biped, w = row.wheeled;
      const fmt = (r) => `${pct.format(r.failure_rate)} (≤ ${pct.format(r.failure_rate_upper_95)})`;
      tr.append(cell(job.charAt(0).toUpperCase() + job.slice(1)), cell(fmt(b), "num" + (b.failure_rate > 0.05 ? " bad" : "")),
        cell(fmt(w), "num" + (w.failure_rate > 0.05 ? " bad" : "")), cell(WHY[job] || ""));
      tb.appendChild(tr);
    });
  }).catch(() => { $("#jobs-table tbody").innerHTML = "<tr><td colspan=4>Run python -m breakpoint humanoid to produce this table.</td></tr>"; });

  // ---------- design levers (from python -m breakpoint design) ----------
  fetch("/assets/data/levers.json").then((r) => r.json()).then((d) => {
    const tb = $("#levers-table tbody");
    const most = Math.max(1, ...d.levers.map((l) => l.fixed + l.broken));
    const n = (v) => Math.round(v);
    d.levers.forEach((l) => {
      const tr = document.createElement("tr");
      const td = (txt, cls) => { const c = document.createElement("td"); c.textContent = txt; if (cls) c.className = cls; return c; };
      const flips = td("", "num flips");
      const bar = document.createElement("span"); bar.className = "flip-bar"; bar.setAttribute("aria-hidden", "true");
      const good = document.createElement("span"); good.className = "flip-fixed"; good.style.width = `${(100 * l.fixed) / most}%`;
      const bad = document.createElement("span"); bad.className = "flip-broken"; bad.style.width = `${(100 * l.broken) / most}%`;
      bar.append(good, bad);
      flips.append(`${l.fixed} / ${l.broken}`, bar);
      const env = l.envelope, base = d.baseline.envelope;
      const delta = (k) => (n(env[k]) === n(base[k]) ? `${n(env[k])}` : `${n(env[k])} (${env[k] > base[k] ? "+" : "−"}${Math.abs(n(env[k]) - n(base[k]))})`);
      tr.append(td(l.title), td(l.owner), flips, td(l.p_value < 0.001 ? "< 0.001" : l.p_value.toFixed(3), "num" + (l.p_value < 0.05 ? " good" : "")),
        td(`${delta("left")} / ${delta("front")} N·s`, "num"));
      tb.appendChild(tr);
    });
    const top = d.levers[0];
    const lost = top.broken ? `${top.broken} lost` : "none lost";
    const ankle = d.levers.find((l) => l.key === "ankle_torque");
    const stance = d.levers.find((l) => l.key === "stance_width");
    let note = `Biggest single win: ${top.title.toLowerCase()} (${top.fixed} shifts saved, ${lost}).`;
    if (ankle && ankle.fixed + ankle.broken === 0) note += " Ankle torque is a control that can’t move: in every job the foot edge, not the ankle, limits the centre of pressure.";
    if (stance && stance.fixed && stance.broken) note += ` A wider stance cuts both ways (${stance.fixed} saved, ${stance.broken} lost): the loaded foot sits further out during the swing.`;
    $("#levers-note").textContent = note;
  }).catch(() => { $("#levers-table tbody").innerHTML = "<tr><td colspan=5>Run python -m breakpoint design to produce this table.</td></tr>"; });

  // ---------- wire up ----------
  const outs = { imp: (v) => `${v} N·s`, payload: (v) => `${v} kg`, friction: (v) => `μ ${(+v).toFixed(2)}`, latency: (v) => `${v} ms`,
    strength: (v) => `${Math.round(v * 100)}%`, step: (v) => `${Math.round(v * 100)} cm` };
  Object.keys(outs).forEach((id) => {
    const el = $("#" + id);
    el.addEventListener("input", () => { $(`#${id}-out`).textContent = outs[id](el.value); if (id !== "imp") refreshEnvelope(); });
  });
  document.querySelectorAll('input[name="body"]').forEach((r) => r.addEventListener("change", () => { shove(last.dir); drawCapMap(); }));
  document.querySelectorAll(".dir-pad button").forEach((b) => b.addEventListener("click", () => shove(+b.dataset.d)));
  canvas.addEventListener("click", (e) => {
    const r = canvas.getBoundingClientRect();
    const px = (e.clientX - r.left) * canvas.width / r.width, py = (e.clientY - r.top) * canvas.height / r.height;
    const [cx, cy] = shownCom || toScreen(0, 0, { sx: 0, sy: 0 });
    // click position in the robot frame (x forward = up, y left = screen left); the shove travels from it to the robot
    const X = -(py - cy), Y = -(px - cx);
    const dir = Math.round(Math.atan2(-Y, -X) * 180 / Math.PI);
    shove(dir);
  });
  canvas.addEventListener("keydown", (e) => {
    const map = { ArrowLeft: 90, ArrowRight: -90, ArrowUp: 0, ArrowDown: 180 }; // arrow = way the robot is pushed
    if (e.key in map) { e.preventDefault(); shove(map[e.key]); }
  });
  drawFrame({ com: [], capture: [], footsteps: [] }, 0, { sx: 0, sy: 0 });
  refreshEnvelope();
  shove(-90);
})();
