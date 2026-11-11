// Breakpoint lab: build a world, run it, shove it, probe it, check its odds.
(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const num1 = new Intl.NumberFormat("en", { maximumFractionDigits: 1 });
  const num2 = new Intl.NumberFormat("en", { maximumFractionDigits: 2 });
  const pct = new Intl.NumberFormat("en", { style: "percent", maximumFractionDigits: 1 });
  const DIR_NAMES = { "-90": "left", "90": "right", "180": "front", "0": "rear", "-45": "rear-left", "45": "rear-right", "-135": "front-left", "135": "front-right" };
  const KEY_DIRS = { ArrowLeft: 90, ArrowRight: -90, ArrowUp: 0, ArrowDown: 180 }; // arrow = the way the robot is pushed
  const MAX_IMAGE = 3_000_000;

  const state = { scenario: null, run: null, ghost: null, view: null };
  const url = new URL(location.href);

  async function api(path, body) {
    const res = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
    let data = null;
    try { data = await res.json(); } catch (e) { data = null; }
    if (!res.ok) {
      const detail = data && (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
      throw new Error(detail || `The server answered ${res.status}. Try again in a moment.`);
    }
    return data;
  }

  function busy(btn, on, label) {
    btn.disabled = on;
    if (on) { btn.dataset.label = btn.textContent; btn.textContent = label; } else if (btn.dataset.label) btn.textContent = btn.dataset.label;
    document.body.classList.toggle("is-busy", on);
  }

  function syncUrl() {
    url.searchParams.set("q", $("#desc").value.slice(0, 300));
    url.searchParams.set("policy", $("#policy").value);
    if ($("#ghost").value) url.searchParams.set("ghost", $("#ghost").value); else url.searchParams.delete("ghost");
    history.replaceState(null, "", url);
  }

  // ---------- options ----------
  async function loadOptions() {
    const fallbackPolicies = [{ name: "gated-v2field", label: "Best · gated v2field" }, { name: "cpg", label: "Bare gait · no learning" }];
    let policies = fallbackPolicies, profiles = [], axes = [];
    try { policies = (await api("/api/policies")).policies; } catch (e) { /* keep fallback list */ }
    try { profiles = (await api("/api/profiles")).profiles; } catch (e) { profiles = [{ name: "warehouse" }, { name: "sidewalk" }, { name: "home" }, { name: "construction" }, { name: "teleop" }]; }
    try { axes = (await api("/api/axes")).axes; } catch (e) { axes = [{ key: "push_left", label: "Push from left" }]; }
    const fill = (sel, items, text) => items.forEach((it) => { const o = document.createElement("option"); o.value = it.name || it.key; o.textContent = text(it); sel.appendChild(o); });
    fill($("#policy"), policies, (p) => p.label + (p.score ? ` · ${p.score}` : ""));
    fill($("#ghost"), policies, (p) => p.label);
    fill($("#profile"), profiles, (p) => p.name.charAt(0).toUpperCase() + p.name.slice(1));
    fill($("#axis"), axes, (a) => a.label);
    const qp = url.searchParams.get("policy");
    if (qp && policies.some((p) => p.name === qp)) $("#policy").value = qp;
    const qg = url.searchParams.get("ghost");
    if (qg && policies.some((p) => p.name === qg)) $("#ghost").value = qg;
  }

  // ---------- scenario ----------
  function renderScenario(source, matched) {
    const s = state.scenario;
    const shown = Object.assign({}, s); delete shown.notes; delete shown.name;
    $("#scenario-json").textContent = JSON.stringify(shown, null, 2);
    const bits = [s.terrain, `μ ${num2.format(s.friction)}`];
    if (s.payload_kg) bits.push(`${num1.format(s.payload_kg)} kg`);
    if (s.latency_steps) bits.push(`${s.latency_steps * 20} ms lag`);
    $("#scenario-summary").textContent = "· " + bits.join(" · ");
    if (source) $("#compile-status").textContent = `Built by ${source === "rules" ? "the offline parser" : source}${matched && matched.length ? ": " + matched.join(", ") : ""}.`;
    renderShoves();
  }

  function renderShoves() {
    const list = $("#shove-list");
    list.replaceChildren();
    const pushes = (state.scenario && state.scenario.pushes) || [];
    if (!pushes.length) {
      const li = document.createElement("li"); li.className = "empty hint"; li.textContent = "No shoves yet. Add one, or press an arrow key while it plays.";
      list.appendChild(li); return;
    }
    pushes.forEach((p, i) => {
      const li = document.createElement("li");
      const span = document.createElement("span");
      const dir = DIR_NAMES[String(Math.round(p.direction_deg))] || `${Math.round(p.direction_deg)}°`;
      span.textContent = `${num1.format(p.force * p.duration)} N·s from the ${dir} at ${num1.format(p.t)} s`;
      const rm = document.createElement("button");
      rm.type = "button"; rm.className = "shove-remove"; rm.textContent = "Remove";
      rm.setAttribute("aria-label", `Remove the shove at ${num1.format(p.t)} seconds`);
      rm.addEventListener("click", () => { state.scenario.pushes.splice(i, 1); renderScenario(); });
      li.append(span, rm);
      list.appendChild(li);
    });
  }

  async function compile(ev) {
    if (ev) ev.preventDefault();
    const btn = $("#compile-btn");
    busy(btn, true, "Building…");
    try {
      const r = await api("/api/compile", { text: $("#desc").value });
      state.scenario = r.scenario;
      renderScenario(r.source, r.matched);
      syncUrl();
    } catch (e) { $("#compile-status").textContent = `Couldn’t build the world: ${e.message}`; }
    finally { busy(btn, false); }
  }

  async function fromPhoto(file) {
    if (!file) return;
    if (file.size > MAX_IMAGE) { $("#compile-status").textContent = "That photo is over 3 MB. Choose a smaller one or a JPEG export."; return; }
    $("#compile-status").textContent = "Reading the photo…";
    const b64 = await new Promise((ok, bad) => { const r = new FileReader(); r.onload = () => ok(r.result); r.onerror = bad; r.readAsDataURL(file); });
    try {
      const r = await api("/api/compile_image", { image_b64: b64, caption: $("#desc").value });
      state.scenario = r.scenario;
      renderScenario(r.source === "image-heuristic" ? "the photo’s texture and colour" : r.source, r.matched);
    } catch (e) { $("#compile-status").textContent = `Couldn’t use that photo: ${e.message}`; }
  }

  // ---------- run ----------
  function ensureView() {
    if (state.view || !window.RobotView) return state.view;
    state.view = new window.RobotView($("#viewer"), { background: 0x17161b, ghostColor: 0xf47b25 });
    const scrub = $("#scrub");
    state.view.onFrame = (i, f) => {
      scrub.value = String(i);
      $("#hud").textContent = `t ${num1.format(i * (state.run ? state.run.dt : 0))} s · x ${num2.format(f[0][0])} m`;
    };
    return state.view;
  }

  function verdict(run, ghost) {
    const o = run.outcome;
    const box = $("#verdict");
    box.classList.toggle("pass", o.passed);
    box.classList.toggle("fail", !o.passed);
    $("#verdict-word").textContent = o.passed ? "Passed" : (o.fell ? "Fell" : "Stalled");
    $("#verdict-why").textContent = run.explanation + (ghost ? ` Ghost (${$("#ghost").selectedOptions[0].textContent}): ${ghost.explanation}` : "");
    const rows = [["Distance", `${num2.format(o.distance)} m`], ["Time upright", `${num1.format(o.time_s)} s`],
      ["Mean speed", `${num2.format(o.mean_speed)} m/s`], ["Worst tilt", `${num1.format(Math.acos(Math.min(1, Math.max(-1, o.min_upright))) * 180 / Math.PI)}°`],
      ["Lowest body clearance", `${num1.format(o.min_clearance * 100)} cm`], ["Safety margin", num2.format(o.margin)],
      ["Compute", `${run.compute_ms} ms`]];
    const dl = $("#metrics"); dl.replaceChildren();
    rows.forEach(([k, v]) => { const dt = document.createElement("dt"); dt.textContent = k; const dd = document.createElement("dd"); dd.textContent = v; dl.append(dt, dd); });
  }

  async function run(startFrame) {
    if (!state.scenario) await compile();
    if (!state.scenario) return;
    const btn = $("#run-btn");
    busy(btn, true, "Simulating…");
    try {
      const body = (policy) => ({ policy, seed: 0, scenario: state.scenario });
      const g = $("#ghost").value;
      const [r, gr] = await Promise.all([api("/api/simulate", body($("#policy").value)), g ? api("/api/simulate", body(g)) : Promise.resolve(null)]);
      state.run = r; state.ghost = gr;
      const view = ensureView();
      $("#viewer-empty").hidden = true;
      if (view) {
        view.build(r.specs, r.terrain, gr && gr.specs);
        view.play(r.frames, { dt: r.dt, ghost: gr && gr.frames });
        if (startFrame) { view.showFrame(startFrame); view.resume(); }
        const n = Math.max(r.frames.length, gr ? gr.frames.length : 0);
        const scrub = $("#scrub"); scrub.max = String(n - 1); scrub.disabled = false;
        $("#replay-btn").disabled = false;
        view.onEnd = () => {};
      }
      verdict(r, gr);
      syncUrl();
    } catch (e) {
      $("#verdict-word").textContent = "Error";
      $("#verdict-why").textContent = `The simulation didn’t run: ${e.message}`;
    } finally { busy(btn, false); }
  }

  function keyShove(ev) {
    if (!(ev.key in KEY_DIRS) || !state.run || !state.view) return;
    const tag = (ev.target.tagName || "").toLowerCase();
    if (["input", "textarea", "select"].includes(tag)) return;
    ev.preventDefault();
    const frame = state.view.frame;
    const t = Math.max(0.2, frame * state.run.dt);
    state.scenario.pushes = (state.scenario.pushes || []).concat([{ t: Math.round(t * 10) / 10, force: Number($("#shove-imp").value || 8) / 0.1, direction_deg: KEY_DIRS[ev.key], duration: 0.1 }]).slice(-8);
    renderScenario();
    run(frame);
  }

  // ---------- tools ----------
  async function probe() {
    const btn = $("#probe-btn"); const out = $("#probe-result");
    busy(btn, true, "Bisecting…"); out.textContent = "";
    try {
      const r = await api("/api/probe", { policy: $("#policy").value, axis: $("#axis").value });
      const big = document.createElement("span"); big.className = "big";
      big.textContent = r.fails_nominal ? "Fails with no stress at all" : `${num2.format(r.breaking_point)} ${r.unit}`;
      out.replaceChildren(big, document.createTextNode(`${r.label}: ${r.fraction >= 1 ? "survives the whole tested range" : `${pct.format(r.fraction)} of the tested range`}, ${r.rollouts} rollouts.`));
    } catch (e) { out.textContent = `Couldn’t probe: ${e.message}`; }
    finally { busy(btn, false); }
  }

  async function odds() {
    const btn = $("#odds-btn"); const out = $("#odds-result");
    busy(btn, true, "Sampling…"); out.textContent = "";
    try {
      const r = await api("/api/deploy", { policy: $("#policy").value, profile: $("#profile").value, trials: 12 });
      const big = document.createElement("span"); big.className = "big";
      big.textContent = `${r.failures} of ${r.trials} failed · ${pct.format(r.failure_rate)} (≤ ${pct.format(r.failure_rate_upper_95)})`;
      const causes = Object.entries(r.failures_by_dominant_stressor).map(([k, v]) => `${k} ×${v}`).join(", ");
      const p = document.createElement("span"); p.textContent = `${r.description} ${causes ? "Causes: " + causes + "." : ""}`;
      out.replaceChildren(big, p);
      if (r.worst.length) {
        const ul = document.createElement("ul");
        r.worst.slice(0, 3).forEach((w) => { const li = document.createElement("li"); li.textContent = w.explanation; ul.appendChild(li); });
        out.appendChild(ul);
      }
    } catch (e) { out.textContent = `Couldn’t check the odds: ${e.message}`; }
    finally { busy(btn, false); }
  }

  // ---------- wire up ----------
  document.addEventListener("DOMContentLoaded", async () => {
    const q = url.searchParams.get("q");
    if (q) $("#desc").value = q;
    $("#world-form").addEventListener("submit", compile);
    document.querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => { $("#desc").value = c.textContent; compile(); }));
    $("#photo").addEventListener("change", (e) => fromPhoto(e.target.files[0]));
    $("#shove-add").addEventListener("click", () => {
      if (!state.scenario) return;
      const imp = Number($("#shove-imp").value), t = Number($("#shove-t").value);
      if (!(imp > 0 && imp <= 40) || !(t >= 0.5 && t <= 9)) { $("#compile-status").textContent = "Use a strength from 1 to 40 N·s and a time from 0.5 to 9 s."; return; }
      state.scenario.pushes = (state.scenario.pushes || []).concat([{ t, force: imp / 0.1, direction_deg: Number($("#shove-dir").value), duration: 0.1 }]).slice(-8);
      renderScenario();
    });
    $("#run-btn").addEventListener("click", () => run());
    $("#replay-btn").addEventListener("click", () => { if (state.view && state.run) state.view.play(state.run.frames, { dt: state.run.dt, ghost: state.ghost && state.ghost.frames }); });
    $("#scrub").addEventListener("input", (e) => { if (state.view) { state.view.pause(); state.view.showFrame(Number(e.target.value)); } });
    $("#policy").addEventListener("change", syncUrl);
    $("#ghost").addEventListener("change", syncUrl);
    $("#probe-btn").addEventListener("click", probe);
    $("#odds-btn").addEventListener("click", odds);
    document.addEventListener("keydown", keyShove);
    renderShoves();
    await loadOptions();
    await compile();
    if (state.scenario) run(); // first view shows the robot, not an empty stage
  });
})();
