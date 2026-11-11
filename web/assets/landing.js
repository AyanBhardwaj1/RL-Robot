// Breakpoint landing page: live shove demo, stage slider, findings index, reveals.
(function () {
  "use strict";
  document.documentElement.classList.add("js");
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));

  // Measured push envelope of the best policy (N·s, worst gait phase), robot frame.
  // Force direction: 0 = forward (shove from the rear), +90 = toward its left (shove from the right).
  const ENVELOPE = [
    ["front", 180, 13.75], ["front-right", 135, 8.125], ["right", 90, 6.875], ["rear-right", 45, 6.25],
    ["rear", 0, 8.125], ["rear-left", -45, 5.625], ["left", -90, 5.625], ["front-left", -135, 8.75],
  ];
  const LIMITS = { left: 5.625, right: 6.875, front: 13.75, rear: 8.125 };
  const DIRS = { left: -90, right: 90, front: 180, rear: 0 };
  const PUSH_T = 1.2;
  const fmt = new Intl.NumberFormat("en", { maximumFractionDigits: 1 });

  // ---- Envelope motif: the robot faces up; a shove from the left is drawn on the left.
  (function drawEnvelope() {
    const poly = $("#envelope-shape");
    if (!poly) return;
    const max = Math.max(...ENVELOPE.map((e) => e[2]));
    const pts = ENVELOPE.map(([, deg, v]) => {
      const a = (deg * Math.PI) / 180;
      const r = 28 + 72 * (v / max);
      return `${(Math.sin(a) * r).toFixed(1)},${(Math.cos(a) * r).toFixed(1)}`;
    });
    poly.setAttribute("points", pts.join(" "));
  })();

  // ---- Footer block wordmark: letter heights follow the best policy's breaking-point fractions.
  (function blockmark() {
    const el = $("#blockmark");
    if (!el) return;
    const fractions = [1, 0.94, 1, 0.61, 1, 0.98, 1, 1, 0.34, 0.2];
    "Breakpoint".split("").forEach((ch, i) => {
      const s = document.createElement("span");
      s.textContent = ch;
      s.setAttribute("aria-hidden", "true");
      s.style.height = `${60 + 40 * fractions[i]}%`;
      el.appendChild(s);
    });
    const y = $("#year");
    if (y) y.textContent = String(new Date().getFullYear());
  })();

  // ---- Reveal each section's parts once, staggered.
  (function reveal() {
    const parts = "h2, .how-note, .stat-tile, .odds-tile, .index > li, .caught-frame, .sheet, .clip, .cert-side > p, .cert-side > a, .tile, details";
    $$("[data-reveal]").forEach((sec) => {
      $$(parts, sec).forEach((el, i) => { el.classList.add("rv"); el.style.setProperty("--i", String(Math.min(i, 8))); });
    });
    if (reduced || !("IntersectionObserver" in window)) {
      $$("[data-reveal]").forEach((s) => s.classList.add("is-in"));
      return;
    }
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("is-in"); io.unobserve(e.target); } });
    }, { rootMargin: "0px 0px -12% 0px" });
    $$("[data-reveal]").forEach((s) => io.observe(s));
  })();

  // ---- How it hunts: one slider drives the clause and the interface fragment.
  (function stages() {
    const range = $("#how-range");
    if (!range) return;
    const clauses = [
      ["Probe every stress", "17 stresses, from a shove at 4 points in the stride to 120 ms of lag. Each one is pushed until the robot stops passing, then halved back to the exact edge."],
      ["Hunt the mildest fall", "A cross-entropy search scores failing worlds by how mild they are, and keeps the gentlest failure of each kind in an atlas. Each one is explained in plain words."],
      ["Shrink it to a test", "Every failure is bisected back toward a normal day until it barely fails, then saved as a JSON test case with its seed. Your robot’s failures become a regression suite."],
      ["Harden, then prove it", "PPO replays the failures and the worlds you will deploy to. A checkpoint ships only if it beats the start, judged without training noise."],
    ];
    const clause = $(".how-clause");
    const note = $("#how-note");
    const frags = $$(".frag");
    const stops = $$(".how-stops li");
    function set(i) {
      clause.textContent = clauses[i][0];
      note.textContent = clauses[i][1];
      frags.forEach((f, k) => f.classList.toggle("is-on", k === i));
      stops.forEach((s, k) => s.classList.toggle("is-on", k === i));
      range.setAttribute("aria-valuetext", stops[i].textContent);
    }
    range.addEventListener("input", () => set(Number(range.value)));
    stops.forEach((s, k) => s.addEventListener("click", () => { range.value = String(k); set(k); }));
    set(0);
  })();

  // ---- What it caught: index with preview (the signature move).
  (function index() {
    const img = $("#caught-img");
    if (!img) return;
    const rows = $$(".index-row");
    let current = rows.find((r) => r.classList.contains("is-on"));
    function show(row) {
      if (row === current) return;
      rows.forEach((r) => r.classList.toggle("is-on", r === row));
      const src = row.parentElement.querySelector(".row-img");
      current = row;
      if (reduced) { img.src = src.src; img.alt = src.alt; return; }
      img.classList.add("is-swapping");
      setTimeout(() => { img.src = src.getAttribute("src"); img.alt = src.alt; img.classList.remove("is-swapping"); }, 120);
    }
    rows.forEach((r) => {
      r.addEventListener("mouseenter", () => show(r));
      r.addEventListener("focus", () => show(r));
      r.addEventListener("click", () => show(r));
    });
  })();

  // ---- The recorded clip: still under reduced motion; tap or Enter toggles it.
  $$(".clip video").forEach((v) => {
    if (reduced) { v.removeAttribute("autoplay"); v.pause(); }
    v.setAttribute("tabindex", "0");
    v.setAttribute("aria-label", "Recorded shove clip. Press Enter to play or pause.");
    const toggle = () => (v.paused ? v.play() : v.pause());
    v.addEventListener("click", toggle);
    v.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } });
  });

  // ---- Hero: the robot walks; the visitor shoves it.
  const resultEl = $("#shove-result");
  let view = null;
  let shoveData = null;

  function decode(frames, quantised) { return window.RobotView.decode(frames, quantised); }

  function startView(hero) {
    const host = $("#hero-robot");
    view = new window.RobotView(host, { background: 0x17161b });
    view.build(hero.specs, null);
    host.classList.add("is-live");
    const walk = decode(hero.frames, hero.quantised);
    if (reduced) { view.play(walk, { dt: hero.dt }); view.pause(); view.showFrame(Math.floor(walk.length / 2)); }
    else view.play(walk, { dt: hero.dt, loop: true });
    const speed = hero.outcome.distance / hero.outcome.time_s;
    resultEl.textContent = `It walks at ${fmt.format(speed)} m/s. Its measured limit from the left is ${fmt.format(LIMITS.left)} N·s.`;
    const pause = $("#hero-pause");
    pause.addEventListener("click", () => {
      const paused = pause.getAttribute("aria-pressed") === "true";
      if (paused) { view.resume(); pause.textContent = "Pause"; pause.setAttribute("aria-pressed", "false"); pause.setAttribute("aria-label", "Pause the robot animation"); }
      else { view.pause(); pause.textContent = "Play"; pause.setAttribute("aria-pressed", "true"); pause.setAttribute("aria-label", "Play the robot animation"); }
    });
  }

  function whenThree(cb) {
    if (window.THREE && window.RobotView) return cb();
    window.addEventListener("load", () => { if (window.THREE && window.RobotView) cb(); });
  }

  whenThree(() => {
    fetch("/assets/data/hero.json").then((r) => r.json()).then(startView).catch(() => {
      resultEl.textContent = "The 3D view could not load. The lab still works: open it from the top right.";
    });
  });

  const impulse = $("#impulse");
  const impulseOut = $("#impulse-out");
  impulse.addEventListener("input", () => { impulseOut.textContent = `${impulse.value} N·s`; });

  async function liveShove(dir, imp) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 15000);
    try {
      const res = await fetch("/api/simulate", {
        method: "POST", headers: { "Content-Type": "application/json" }, signal: ctrl.signal,
        body: JSON.stringify({ policy: "gated-v2field", seed: 0, scenario: { duration_s: 4, phase0_deg: 0,
          pushes: [{ t: PUSH_T, force: imp / 0.1, direction_deg: DIRS[dir], duration: 0.1 }] } }),
      });
      if (!res.ok) throw new Error(String(res.status));
      const d = await res.json();
      return { frames: d.frames, dt: d.dt, quantised: false, outcome: d.outcome, explanation: d.explanation, live: true };
    } finally { clearTimeout(timer); }
  }

  async function recordedShove(dir, imp) {
    if (!shoveData) shoveData = await (await fetch("/assets/data/shoves.json")).json();
    const options = [4, 8, 12];
    const near = options.reduce((a, b) => (Math.abs(b - imp) < Math.abs(a - imp) ? b : a));
    const r = shoveData.replays[`${dir}-${near}`];
    return Object.assign({ live: false, recordedImpulse: near }, r);
  }

  $("#shove-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const btn = $("#shove-btn");
    const dir = new FormData(ev.target).get("dir");
    const imp = Number(impulse.value);
    btn.disabled = true;
    btn.textContent = "Simulating…";
    resultEl.textContent = `Shoving from the ${dir} with ${imp} N·s…`;
    let r;
    try { r = await liveShove(dir, imp); } catch (e) {
      try { r = await recordedShove(dir, imp); } catch (e2) { r = null; }
    }
    btn.disabled = false;
    btn.textContent = "Shove the robot";
    if (!r) { resultEl.textContent = "The simulator didn’t answer. Try again, or open the lab."; return; }
    if (view) view.play(decode(r.frames, r.quantised), { dt: r.dt, loop: false });
    const verdict = r.outcome.fell ? "It fell." : (r.outcome.passed ? "It held." : "It stayed up but stalled.");
    const limit = LIMITS[dir];
    const note = r.live ? "" : ` (Pre-recorded at ${r.recordedImpulse} N·s: the live simulator was busy.)`;
    resultEl.textContent = `${verdict} ${r.explanation} Measured limit from the ${dir}: ${fmt.format(limit)} N·s.${note}`;
  });
})();
