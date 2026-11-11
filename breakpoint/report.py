"""Readiness scorecard + self-contained HTML report (inline SVG, no external assets)."""

from __future__ import annotations

import html
import math
from datetime import datetime, timezone

from .evaluate import failure_upper_bound
from .explain import explain
from .probe import PUSH_DIRS
from .scenario import Scenario

GRADES = ((85, "FIELD-READY", "Cleared for unsupervised field trials in the tested envelope."),
          (70, "SUPERVISED", "Cleared for supervised trials with a spotter and kill switch."),
          (50, "LAB-ONLY", "Keep on the tether: known failure modes inside plausible conditions."),
          (0, "NOT READY", "Fails under mild, realistic stress. Harden before any hardware time."))


def scorecard(probe: list[dict], nominal: list[dict], atlas: dict[str, dict] | None) -> dict:
    push = [r["fraction"] for r in probe if r["key"].startswith("push_")]
    other = [r["fraction"] for r in probe if not r["key"].startswith("push_")]
    margin = 0.4 * (sum(push) / len(push) if push else 0) + 0.6 * (sum(other) / len(other) if other else 0)
    n = len(nominal)
    fails = sum(1 for r in nominal if not r["passed"])
    pass_rate = (n - fails) / n if n else 0.0
    mild = [c for c in (atlas or {}).values() if c["severity"] <= 0.5]
    adversarial = 1.0 - min(1.0, len(mild) / 5.0)
    score = round(100 * (0.45 * margin + 0.4 * pass_rate + 0.15 * adversarial), 1)
    grade = next(g for g in GRADES if score >= g[0])
    return {
        "score": score, "grade": grade[1], "verdict": grade[2],
        "margin_index": round(margin, 3), "nominal_pass_rate": round(pass_rate, 3),
        "nominal_trials": n, "nominal_failures": fails,
        "failure_rate_upper_95": round(failure_upper_bound(n, fails), 4),
        "mild_failure_modes": len(mild),
        "weakest_axes": [r["label"] for r in sorted(probe, key=lambda r: r["fraction"])[:3]],
    }


# ------------------------------------------------------------------ svg
def _polar(series: dict[str, dict[str, float]], size: int = 300, vmax: float | None = None) -> str:
    peak = max((v for vals in series.values() for v in vals.values()), default=0.0)
    vmax = vmax or max(5.0, 5.0 * math.ceil(peak * 1.15 / 5.0))
    c = size / 2
    rad = c - 34
    order = ["front", "front-left", "left", "rear-left", "rear", "rear-right", "right", "front-right"]
    parts = [f'<svg viewBox="0 0 {size} {size}" class="chart" role="img" aria-label="Push recovery by direction">']
    for frac in (0.25, 0.5, 0.75, 1.0):
        parts.append(f'<circle cx="{c}" cy="{c}" r="{rad * frac:.1f}" class="grid"/>')
        parts.append(f'<text x="{c + 3}" y="{c - rad * frac + 11:.1f}" class="tick">{vmax * frac:.0f}</text>')
    for name in order:
        ang = math.radians(PUSH_DIRS[name])
        # screen: robot faces up. A push *from* the front comes from the top.
        sx, sy = -math.sin(ang), -math.cos(ang)
        parts.append(f'<line x1="{c}" y1="{c}" x2="{c + rad * sx:.1f}" y2="{c - rad * sy:.1f}" class="grid"/>')
        parts.append(f'<text x="{c + (rad + 18) * sx:.1f}" y="{c - (rad + 18) * sy + 4:.1f}" class="lbl" text-anchor="middle">{name}</text>')
    for i, (label, vals) in enumerate(series.items()):
        pts = []
        for name in order:
            ang = math.radians(PUSH_DIRS[name])
            r = rad * min(1.0, vals.get(name, 0.0) / vmax)
            pts.append(f"{c - r * math.sin(ang):.1f},{c + r * math.cos(ang):.1f}")
        parts.append(f'<polygon points="{" ".join(pts)}" class="s{i}"><title>{html.escape(label)}</title></polygon>')
    parts.append("</svg>")
    return "".join(parts)


def _bars(probes: dict[str, list[dict]]) -> str:
    labels = [r for r in next(iter(probes.values())) if not r["key"].startswith("push_")]
    row_h, w, lw = 30, 640, 150
    h = row_h * len(labels) + 10
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart wide" role="img" aria-label="Breaking points">']
    k = len(probes)
    for j, r in enumerate(labels):
        y = 6 + j * row_h
        out.append(f'<text x="{lw - 8}" y="{y + 15}" class="lbl" text-anchor="end">{html.escape(r["label"])}</text>')
        out.append(f'<rect x="{lw}" y="{y + 2}" width="{w - lw - 90}" height="{row_h - 8}" class="track"/>')
        for i, (name, res) in enumerate(probes.items()):
            rr = next(x for x in res if x["key"] == r["key"])
            bw = (w - lw - 90) * rr["fraction"]
            bh = (row_h - 8) / k
            out.append(f'<rect x="{lw}" y="{y + 2 + i * bh:.1f}" width="{bw:.1f}" height="{bh - 1:.1f}" class="s{i}">'
                       f'<title>{html.escape(name)}: {rr["breaking_point"]} {html.escape(rr["unit"])}</title></rect>')
        last = probes[list(probes)[-1]]
        rr = next(x for x in last if x["key"] == r["key"])
        out.append(f'<text x="{w - 84}" y="{y + 15}" class="val">{rr["breaking_point"]:g} {html.escape(rr["unit"])}</text>')
    out.append("</svg>")
    return "".join(out)


CSS = """
:root{--bg:#f6f5f1;--fg:#1d1d1f;--muted:#6b6b70;--card:#fff;--line:#e3e1da;--s0:#9aa0a6;--s1:#1c7ed6;--s2:#7048e8;--s3:#e8590c;--ok:#2b8a3e;--warn:#e67700;--bad:#c92a2a}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#121214;--fg:#ececf0;--muted:#9a9aa3;--card:#1c1c20;--line:#2c2c33;--s0:#6c7178;--s1:#4dabf7;--s2:#9775fa;--s3:#ff7a2e}}
:root[data-theme=dark]{--bg:#121214;--fg:#ececf0;--muted:#9a9aa3;--card:#1c1c20;--line:#2c2c33;--s0:#6c7178;--s1:#4dabf7;--s2:#9775fa;--s3:#ff7a2e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 64px}h1{font-size:28px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 12px}
.muted{color:var(--muted)}.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.score{font-size:56px;font-weight:700;line-height:1}.badge{display:inline-block;padding:3px 10px;border-radius:99px;font-weight:600;font-size:13px;color:#fff}
.chart{width:100%;height:auto}.chart .grid{fill:none;stroke:var(--line)}.chart .tick,.chart .lbl,.chart .val{fill:var(--muted);font-size:11px}
.chart .val{fill:var(--fg)}.chart .track{fill:var(--line);opacity:.5}.s0{fill:var(--s0);fill-opacity:.35;stroke:var(--s0);stroke-width:2}
.s1{fill:var(--s1);fill-opacity:.35;stroke:var(--s1);stroke-width:2}.s2{fill:var(--s2);fill-opacity:.25;stroke:var(--s2);stroke-width:2}
.s3{fill:var(--s3);fill-opacity:.25;stroke:var(--s3);stroke-width:2}rect.s0,rect.s1,rect.s2,rect.s3{fill-opacity:.9;stroke:none}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}
.key{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}code{font-size:12px}
.tablewrap{overflow-x:auto}
"""


def _heatmap(bm: dict) -> str:
    xs, ys, grid = bm["x"]["values"], bm["y"]["values"], bm["pass_rate"]
    cw, ch, ml, mb = 56, 34, 70, 40
    w, h = ml + cw * len(xs) + 10, ch * len(ys) + mb + 10
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" style="max-width:520px" role="img" aria-label="Failure boundary">']
    for j, yv in enumerate(ys):
        y = 5 + (len(ys) - 1 - j) * ch
        out.append(f'<text x="{ml - 6}" y="{y + ch / 2 + 4}" class="lbl" text-anchor="end">{yv:g}</text>')
        for i, xv in enumerate(xs):
            r = grid[j][i]
            out.append(f'<rect x="{ml + i * cw}" y="{y}" width="{cw - 2}" height="{ch - 2}" rx="3" '
                       f'style="fill:{"var(--ok)" if r >= 1 else ("var(--warn)" if r > 0 else "var(--bad)")};fill-opacity:{0.35 + 0.5 * (1 - r) if r < 1 else 0.55}">'
                       f'<title>{html.escape(bm["x"]["label"])} {xv:g}, {html.escape(bm["y"]["label"])} {yv:g}: {r:.0%} pass</title></rect>')
    for i, xv in enumerate(xs):
        out.append(f'<text x="{ml + i * cw + cw / 2}" y="{h - mb + 18}" class="lbl" text-anchor="middle">{xv:g}</text>')
    out.append(f'<text x="{ml + cw * len(xs) / 2}" y="{h - 6}" class="lbl" text-anchor="middle">{html.escape(bm["x"]["label"])} ({html.escape(bm["x"]["unit"])})</text>')
    out.append("</svg>")
    return "".join(out)


ATLAS_ROWS = ("push", "low friction", "payload", "weak motors", "latency", "sensor noise", "terrain")
ATLAS_COLS = ("≤0.25", "≤0.5", "≤0.75", "≤1.0", ">1.0")


def _atlas_grid(atlas: dict[str, dict] | None) -> str:
    """MAP-Elites view: which (stressor, severity band) cells contain a known failure."""
    if not atlas:
        return ""
    cw, ch, ml = 64, 26, 110
    w, h = ml + cw * len(ATLAS_COLS) + 10, ch * (len(ATLAS_ROWS) + 1) + 10
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" style="max-width:560px" role="img" aria-label="Failure atlas grid">']
    for i, col in enumerate(ATLAS_COLS):
        out.append(f'<text x="{ml + i * cw + cw / 2}" y="16" class="lbl" text-anchor="middle">{col}</text>')
    for j, row in enumerate(ATLAS_ROWS):
        y = 24 + j * ch
        out.append(f'<text x="{ml - 8}" y="{y + ch / 2 + 2}" class="lbl" text-anchor="end">{row}</text>')
        for i in range(len(ATLAS_COLS)):
            cell = atlas.get(f"{row}|{i}")
            style = "fill:var(--bad);fill-opacity:.7" if cell else "fill:var(--line);fill-opacity:.6"
            tip = f"{row}, severity {ATLAS_COLS[i]}: " + (f"fails ({cell['scenario']['name']})" if cell else "no failure found")
            out.append(f'<rect x="{ml + i * cw}" y="{y}" width="{cw - 3}" height="{ch - 3}" rx="3" style="{style}">'
                       f'<title>{html.escape(tip)}</title></rect>')
    out.append("</svg>")
    return ('<p class="muted">MAP-Elites view: red = a failure was found in that (dominant stressor × severity) '
            f'cell; grey = none within the search budget.</p><div class="card">{"".join(out)}</div>')


def _compound_section(extras: dict | None, who: str = "") -> str:
    if not extras:
        return ""
    parts = [(f'<h2>Compound stress · {html.escape(who)}</h2><p class="muted">Pairs of the weakest axes, each set to a fraction '
              'of its own breaking point. Single-axis margins cannot see these failures.</p><div class="card tablewrap"><table>'
              '<tr><th>Stressor A</th><th>Stressor B</th><th>Each at</th><th>Pass rate</th></tr>')]
    for c in extras.get("interactions", []):
        flag = ' style="color:var(--bad);font-weight:600"' if c["compound_failure"] else ""
        parts.append(f"<tr><td>{html.escape(c['a'])}</td><td>{html.escape(c['b'])}</td><td>{c['fraction']:.0%}</td>"
                     f"<td{flag}>{c['pass_rate']:.0%}</td></tr>")
    parts.append("</table></div>")
    bm = extras.get("boundary")
    if bm:
        parts.append(f'<h2>Failure boundary</h2><p class="muted">{html.escape(bm["y"]["label"])} ({html.escape(bm["y"]["unit"])}, rows) '
                     f'× {html.escape(bm["x"]["label"])} (columns). Green passes, red falls.</p><div class="card">{_heatmap(bm)}</div>')
    rt = extras.get("redteam")
    if rt:
        parts.append(f'<h2>Red team</h2><p class="muted">Realistic multi-stressor situations proposed by the '
                     f'{"LLM red-teamer" if rt["source"] == "llm" else "offline red-team templates"} from the measured weak spots.</p>'
                     '<div class="card tablewrap"><table><tr><th>Situation</th><th>Pass rate (3 seeds)</th></tr>')
        for a in rt["attacks"]:
            flag = ' style="color:var(--bad);font-weight:600"' if a["pass_rate"] < 1 else ""
            parts.append(f"<tr><td>{html.escape(a['scenario']['name'])}</td><td{flag}>{a['pass_rate']:.0%}</td></tr>")
        parts.append("</table></div>")
    return "".join(parts)


def render_html(title: str, cards: dict[str, dict], probes: dict[str, list[dict]],
                atlas: dict[str, dict] | None, hunt_meta: dict | None, notes: list[str] | None = None,
                extras: dict | None = None) -> str:
    names = list(probes)
    best = max(names, key=lambda n: cards[n]["score"])
    final = cards[best]
    color = {"FIELD-READY": "var(--ok)", "SUPERVISED": "var(--ok)", "LAB-ONLY": "var(--warn)"}.get(final["grade"], "var(--bad)")
    legend = "".join(f'<span class="muted" style="margin-right:14px"><span class="key s{i}" '
                     f'style="background:var(--s{i})"></span>{html.escape(n)}</span>' for i, n in enumerate(names))
    rows = []
    for n in names:
        c = cards[n]
        rows.append(f"<tr><td>{html.escape(n)}</td><td><b>{c['score']}</b></td><td>{c['grade']}</td>"
                    f"<td>{c['margin_index']}</td><td>{c['nominal_pass_rate']:.0%} ({c['nominal_trials']} runs)</td>"
                    f"<td>≤ {c['failure_rate_upper_95']:.1%}</td><td>{c['mild_failure_modes']}</td></tr>")
    atlas_rows = []
    for cell in sorted((atlas or {}).values(), key=lambda c: c["severity"]):
        s = cell["scenario"]
        push = s["pushes"][0] if s["pushes"] else None
        desc = [f"μ={s['friction']:.2f}", f"payload {s['payload_kg']:.1f} kg", f"motors ×{s['motor_strength']:.2f}",
                f"lag {int(s['latency_steps']) * 20} ms", f"noise σ {s['obs_noise']:.2f}", f"slope {s['slope_deg']:.0f}°",
                f"rough {s['roughness'] * 100:.1f} cm"]
        if push:
            desc.append(f"push {push['force'] * push['duration']:.1f} N·s @ {push['direction_deg']:.0f}°")
        o = cell["outcome"]
        why = explain(Scenario.from_dict(s), o)
        atlas_rows.append(f"<tr><td>{html.escape(cell['stressor'])}</td><td>{cell['severity']:.2f}</td>"
                          f"<td>{html.escape(why)}</td>"
                          f"<td><code>{html.escape(', '.join(desc))}</code></td></tr>")
    budget = ""
    if hunt_meta:
        budget = (f"<p class='muted'>Search budget: {hunt_meta['evaluations']} adversarial rollouts, "
                  f"{hunt_meta['failures_found']} failures, {len(atlas or {})} distinct cells. "
                  "No failure found in a cell does not prove absence; it bounds what this budget could find.</p>")
    notes_html = "".join(f"<li>{html.escape(n)}</li>" for n in (notes or []))
    polar = _polar({n: {r["key"][5:]: r["breaking_point"] for r in probes[n] if r["key"].startswith("push_")} for n in names})
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>
<p class="muted">Breakpoint readiness report · {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>
<h1>{html.escape(title)}</h1>
<div class="grid2" style="margin-top:20px">
 <div class="card"><div class="muted">Readiness score · {html.escape(best)}{" (best)" if len(names) > 1 else ""}</div>
  <div class="score">{final['score']}</div>
  <p><span class="badge" style="background:{color}">{final['grade']}</span></p><p>{html.escape(final['verdict'])}</p></div>
 <div class="card"><div class="muted">Weakest axes</div><ol>{''.join(f'<li>{html.escape(a)}</li>' for a in final['weakest_axes'])}</ol>
  <div class="muted">Nominal failure rate (95% upper bound)</div><div style="font-size:24px;font-weight:600">≤ {final['failure_rate_upper_95']:.1%}</div></div>
</div>
<h2>Scorecard</h2><div class="card tablewrap"><table><tr><th>Policy</th><th>Score</th><th>Grade</th><th>Margin index</th><th>Nominal pass</th><th>Fail-rate UB</th><th>Mild failure modes</th></tr>{''.join(rows)}</table></div>
<h2>Push recovery envelope</h2><p class="muted">Largest recoverable 0.1 s torso impulse per direction (N·s), bisected over gait phases. {legend}</p>
<div class="grid2"><div class="card">{polar}</div><div class="card"><p class="muted">How to read: the robot faces up. A dent in the shape is a direction where a shove topples it. Side pushes are usually the weakest because a trotting quadruped has a narrow support polygon laterally.</p></div></div>
<h2>Breaking points</h2><p class="muted">Bar = fraction of the tested range survived on every seed. {legend}</p><div class="card">{_bars(probes)}</div>
<h2>Failure atlas</h2>{budget}<div class="card tablewrap"><table><tr><th>Mode</th><th>Severity</th><th>Outcome</th><th>Mildest world that breaks it</th></tr>{''.join(atlas_rows) or '<tr><td colspan=4>No failures found.</td></tr>'}</table></div>
{_atlas_grid(atlas)}
{_compound_section(extras, names[-1])}
<h2>Method</h2><div class="card"><ul>
<li>Score = 45% margin index (push envelope 40%, other stressors 60%) + 40% nominal pass rate under domain-randomised conditions + 15% adversarial (penalises failure modes found at severity ≤ 0.5).</li>
<li>Pass = no fall (tilt &gt; 65° or body clearance &lt; 11 cm) and ≥ 35% of commanded distance covered.</li>
<li>Failure-rate bound: exact binomial with zero failures (≈3/n), Wilson score otherwise.</li>{notes_html}</ul></div>
</main></body></html>"""
