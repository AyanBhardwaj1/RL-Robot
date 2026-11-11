// Teleop demonstration filter in the browser: a port of breakpoint/episodes.py on top of humanoid.js.
// tests/test_humanoid_js.py checks the tags against Python on the sample log.
(function (root) {
  "use strict";
  const H = typeof module !== "undefined" && module.exports ? require("./humanoid.js") : root.Humanoid;
  const STALL_MS = 100, SLOW_MS = 200, NEAR_FALL_MARGIN = 1.25;

  function num(v, what, lo, allowNull) {
    if (v === null && allowNull) return NaN;
    if (typeof v !== "number" || !Number.isFinite(v)) throw new Error(`${what} must be a finite number, got ${JSON.stringify(v)}`);
    if (lo !== undefined && v < lo) throw new Error(`${what} must be ≥ ${lo}, got ${v}`);
    return v;
  }
  // Like Python's dict.get: the default applies only when the key is absent (an explicit null is an error).
  const get = (o, k, d) => (Object.prototype.hasOwnProperty.call(o, k) ? o[k] : d);
  const median = (xs) => { const s = [...xs].sort((a, b) => a - b), n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };

  function trace(ep) {
    const raw = ep.latency_ms;
    if (raw === undefined || raw === null) return [];
    if (!Array.isArray(raw)) throw new Error("latency_ms must be a list of numbers (one per second)");
    return raw.map((x, k) => num(x, `latency_ms[${k}]`, 0, true));
  }

  function latencyAt(tr, t) {
    const finite = tr.filter(Number.isFinite);
    if (!finite.length) return [0, "no latency samples: replayed at 0 ms, an optimistic assumption"];
    const x = tr[Math.min(tr.length - 1, Math.max(0, Math.floor(t)))];
    if (Number.isFinite(x)) return [x, null];
    return [median(finite), `no latency sample at ${t.toFixed(1)} s: used the episode median`];
  }

  function tagEpisode(ep) {
    const name = String(ep.episode ?? "?");
    try { return tag(ep, name); } catch (e) { throw new Error(`episode ${JSON.stringify(name)}: ${e.message}`); }
  }

  function tag(ep, name) {
    const tr = trace(ep);
    const payload = num(get(ep, "payload_kg", 0), "payload_kg", 0), friction = num(get(ep, "friction", 0.8), "friction", 0.01);
    const slope = num(get(ep, "slope_deg", 0), "slope_deg");
    const contacts = ep.contacts || [];
    if (!Array.isArray(contacts)) throw new Error("contacts must be a list");
    const reasons = [];
    let fell = false, near = false;
    contacts.forEach((c, k) => {
      if (!c || typeof c !== "object" || Array.isArray(c)) throw new Error(`contacts[${k}] must be an object with t_s, impulse_ns and direction_deg`);
      const imp = num(get(c, "impulse_ns", 0), `contacts[${k}].impulse_ns`, 0), ang = num(get(c, "direction_deg", 0), `contacts[${k}].direction_deg`);
      const t = num(get(c, "t_s", 0), `contacts[${k}].t_s`, 0);
      const [lat, note] = latencyAt(tr, t);
      if (note) reasons.push(note);
      const cond = { payload_kg: payload, friction, floor_slope_deg: slope, latency_s: lat / 1000 };
      const r = H.recover("biped", imp, ang, cond, false);
      if (!r.recovered) { fell = true; reasons.push(`fall at ${t.toFixed(1)} s: ${Math.round(imp)} N·s shove, ${r.reason}`); }
      else if (!H.recover("biped", imp * NEAR_FALL_MARGIN, ang, cond, false).recovered) {
        near = true;
        reasons.push(`near-fall at ${t.toFixed(1)} s: ${Math.round(imp)} N·s shove recovered in ${r.steps} step(s), 25% harder would fall`);
      }
    });
    let lag = false;
    const finite = tr.filter(Number.isFinite);
    if (finite.length) {
      const peak = Math.max(...finite), med = median(finite);
      if (peak - med > STALL_MS || med > SLOW_MS) { lag = true; reasons.push(`lag: median ${Math.round(med)} ms, peak ${Math.round(peak)} ms`); }
    }
    return { episode: name, tag: fell ? "fall" : near ? "near-fall" : lag ? "lag" : "clean", reasons };
  }

  function parseJsonl(text) {
    const out = [];
    text.split(/\r?\n/).forEach((line, i) => {
      if (!line.trim()) return;
      let rec;
      try { rec = JSON.parse(line); } catch (e) { throw new Error(`line ${i + 1}: not valid JSON`); }
      if (!rec || typeof rec !== "object" || Array.isArray(rec)) throw new Error(`line ${i + 1}: each line must be a JSON object`);
      out.push(rec);
    });
    if (!out.length) throw new Error("no episodes in the file");
    return out;
  }

  function filterEpisodes(eps) {
    const verdicts = eps.map((ep, i) => { try { return tagEpisode(ep); } catch (e) { throw new Error(`record ${i + 1}, ${e.message}`); } });
    const counts = { clean: 0, lag: 0, "near-fall": 0, fall: 0 };
    verdicts.forEach((v) => { counts[v.tag] += 1; });
    return { episodes: verdicts.length, counts, keep_share: verdicts.length ? counts.clean / verdicts.length : 0, verdicts };
  }

  const api = { tagEpisode, filterEpisodes, parseJsonl };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.Episodes = api;
})(typeof window !== "undefined" ? window : globalThis);
