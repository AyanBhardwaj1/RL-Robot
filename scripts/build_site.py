"""Assemble the deployable Vercel site in web/ (committed, so a Vercel Git import works).

    python scripts/build_site.py            # build into web/
    vercel deploy web --prod                # deploy with the CLI, or import the repo in Vercel
                                            # with Root Directory = web

The output holds the static pages from site/, the Python API (site/api), the `breakpoint`
package, the committed policies, the readiness reports, product screenshots, a vendored
three.js, and pre-rendered replays so the hero robot works before the API answers.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from breakpoint.env import CONTROL_DT, QuadrupedEnv
from breakpoint.evaluate import rollout
from breakpoint.explain import explain
from breakpoint.policy import load_policy
from breakpoint.scenario import Scenario

DIST = ROOT / "web"
POLICIES = {
    "policy_v2field.npz": ROOT / "runs/v2field/policy_v2field.npz",
    "policy_v2.npz": ROOT / "runs/v2/policy_v2.npz",
    "policy_hardened.npz": ROOT / "runs/hardened/policy_hardened.npz",
}
REPORTS = {"final": "runs/final/report.html", "leaderboard-v1": "runs/leaderboard/report.html",
           "hardening-v2": "runs/v2field/report.html", "hardening-v1": "runs/hardened/report.html"}
SCREENSHOTS = ["playground.png", "report.png", "report_compound.png", "report_deploy.png", "compare.png", "shove.png"]
SHOVE_DIRS = {"left": -90.0, "right": 90.0, "front": 180.0, "rear": 0.0}
SHOVE_IMPULSES = (4, 8, 12)
PUSH_T = 1.2
FRAME_STRIDE = 2


def quantise(frames: list) -> list:
    """Positions to mm and quaternions to 1/1000 as ints: ~3x smaller JSON, invisible on screen."""
    return [[[round(v * 1000) for v in g] for g in f] for f in frames[::FRAME_STRIDE]]


def replay(policy_spec: str, scn: Scenario) -> dict:
    env = QuadrupedEnv(scn, seed=0)
    out, frames = rollout(load_policy(policy_spec), scn, seed=0, record=True, env=env)
    o = out.to_dict()
    return {"outcome": o, "explanation": explain(scn, o), "frames": quantise(frames), "dt": CONTROL_DT * FRAME_STRIDE,
            "quantised": True}


def build_replays(out_dir: Path) -> None:
    spec = f"gated:{POLICIES['policy_v2field.npz']}"
    env = QuadrupedEnv(Scenario())
    specs = env.geom_specs()
    walk = replay(spec, Scenario(duration_s=5.0, phase0_deg=0.0))
    (out_dir / "hero.json").write_text(json.dumps({"specs": specs, **walk}, separators=(",", ":")))
    shoves = {}
    for name, deg in SHOVE_DIRS.items():
        for imp in SHOVE_IMPULSES:
            scn = Scenario(duration_s=4.0, phase0_deg=0.0).with_(
                pushes=[{"t": PUSH_T, "force": imp / 0.1, "direction_deg": deg, "duration": 0.1}])
            shoves[f"{name}-{imp}"] = replay(spec, scn)
    (out_dir / "shoves.json").write_text(json.dumps({"specs": specs, "push_t": PUSH_T, "replays": shoves},
                                                    separators=(",", ":")))


SITEMAP_PATHS = ("/", "/app", "/humanoid", "/roles", "/docs", "/reports/final")


def write_seo(dist: Path, site_url: str | None) -> None:
    """Sitemaps need absolute URLs: written only when SITE_URL (e.g. https://example.vercel.app) is known."""
    sitemap = dist / "sitemap.xml"
    robots = dist / "robots.txt"
    if not site_url:
        sitemap.unlink(missing_ok=True)
        robots.write_text("User-agent: *\nAllow: /\n")
        return
    base = site_url.rstrip("/")
    urls = "".join(f"  <url><loc>{base}{p}</loc></url>\n" for p in SITEMAP_PATHS)
    sitemap.write_text(f'<?xml version="1.0" encoding="UTF-8"?>\n'
                       f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n{urls}</urlset>\n')
    robots.write_text(f"User-agent: *\nAllow: /\nSitemap: {base}/sitemap.xml\n")
    for page in dist.glob("*.html"):
        page.write_text(re.sub(r'(property="og:image" content=")/', rf"\g<1>{base}/", page.read_text()))


def main(out: Path | None = None, replays: bool = True) -> int:
    DIST = out or globals()["DIST"]
    if DIST.exists():
        shutil.rmtree(DIST)
    shutil.copytree(ROOT / "site", DIST)
    shutil.copytree(ROOT / "breakpoint", DIST / "breakpoint",
                    ignore=shutil.ignore_patterns("__pycache__", "static", "server.py", "cli.py", "__main__.py"))
    (DIST / "policies").mkdir()
    for name, src in POLICIES.items():
        shutil.copy2(src, DIST / "policies" / name)
    (DIST / "reports").mkdir()
    for name, src in REPORTS.items():
        shutil.copy2(ROOT / src, DIST / "reports" / f"{name}.html")
    img = DIST / "assets" / "img"
    img.mkdir(parents=True, exist_ok=True)
    for name in SCREENSHOTS:
        shutil.copy2(ROOT / "docs" / name, img / name)
    shutil.copy2(ROOT / "breakpoint/static/three.min.js", DIST / "assets" / "three.min.js")
    shutil.copy2(ROOT / "breakpoint/static/three.LICENSE", DIST / "assets" / "three.LICENSE.txt")
    data = DIST / "assets" / "data"
    data.mkdir(parents=True, exist_ok=True)
    if replays:
        build_replays(data)
    humanoid = ROOT / "runs" / "humanoid" / "baseline.json"
    if humanoid.is_file():
        shutil.copy2(humanoid, data / "humanoid.json")
    levers = ROOT / "runs" / "humanoid" / "levers.json"
    if levers.is_file():
        shutil.copy2(levers, data / "levers.json")
    write_seo(DIST, os.environ.get("SITE_URL"))
    size = sum(p.stat().st_size for p in DIST.rglob("*") if p.is_file())
    print(f"built {DIST} ({size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
