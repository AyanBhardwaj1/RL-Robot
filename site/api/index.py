"""Breakpoint web API for Vercel (Python serverless, ASGI).

Stateless and size-capped: every request runs a short MuJoCo simulation with the same
`breakpoint` package the CLI uses. Policies are the committed .npz files bundled at build.
"""

from __future__ import annotations

import base64
import binascii
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from breakpoint.compiler import compile_image, compile_text  # noqa: E402
from breakpoint.env import CONTROL_DT, QuadrupedEnv  # noqa: E402
from breakpoint.evaluate import rollout  # noqa: E402
from breakpoint.explain import explain  # noqa: E402
from breakpoint.policy import load_policy  # noqa: E402
from breakpoint.probe import AXIS_BY_KEY, probe_axis  # noqa: E402
from breakpoint.profiles import PROFILES, deployment_check  # noqa: E402
from breakpoint.scenario import Scenario  # noqa: E402
from breakpoint.world import BASE_HEIGHT, HFIELD_HALF_X, HFIELD_HALF_Y, heightfield  # noqa: E402

POLICY_DIR = ROOT / "policies"
MAX_SIM_SECONDS = 10.0
MAX_DEPLOY_TRIALS = 12
MAX_IMAGE_BYTES = 3_000_000  # Vercel request bodies are capped at 4.5 MB
FRAME_STRIDE = 2  # 25 Hz playback
TERRAIN_ROWS, TERRAIN_COLS = 60, 160

POLICIES: dict[str, dict] = {
    "gated-v2field": {"spec": f"gated:{POLICY_DIR / 'policy_v2field.npz'}", "label": "Best: v2field + latency gate",
                      "score": 84.1},
    "v2field": {"spec": str(POLICY_DIR / "policy_v2field.npz"), "label": "v2field (deployment-hardened PPO)",
                "score": 82.6},
    "v2": {"spec": str(POLICY_DIR / "policy_v2.npz"), "label": "v2 (PPO, not hardened)", "score": 73.5},
    "v1-hardened": {"spec": str(POLICY_DIR / "policy_hardened.npz"), "label": "v1 + one hardening round",
                    "score": 77.2},
    "cpg": {"spec": "cpg", "label": "Bare trotting gait (no learning)", "score": 66.8},
}

app = FastAPI(title="Breakpoint API", docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(GZipMiddleware, minimum_size=2000)


def _policy(name: str):
    if name not in POLICIES:
        raise HTTPException(404, f"Unknown policy '{name[:40]}'. Choose one of: {', '.join(POLICIES)}.")
    return load_policy(POLICIES[name]["spec"])


def _scenario(d: dict, max_s: float = MAX_SIM_SECONDS) -> Scenario:
    try:
        scn = Scenario.from_dict(d)
    except ValueError as e:
        raise HTTPException(422, f"Invalid scenario: {e}") from e
    return scn.with_(duration_s=min(scn.duration_s, max_s))


def _terrain(scn: Scenario, seed: int) -> dict:
    import numpy as np

    h = heightfield(scn, seed) + BASE_HEIGHT
    r = np.linspace(0, h.shape[0] - 1, TERRAIN_ROWS).astype(int)
    c = np.linspace(0, h.shape[1] - 1, TERRAIN_COLS).astype(int)
    sub = h[np.ix_(r, c)]
    return {"half_x": HFIELD_HALF_X, "half_y": HFIELD_HALF_Y, "rows": len(r), "cols": len(c),
            "heights": [round(float(v), 3) for v in sub.ravel()], "friction": scn.friction}


class TextReq(BaseModel):
    text: str = Field("", max_length=2000)


class ImageReq(BaseModel):
    image_b64: str = Field(..., max_length=MAX_IMAGE_BYTES * 2)
    caption: str = Field("", max_length=500)


class SimReq(BaseModel):
    scenario: dict
    policy: str = Field("gated-v2field", max_length=40)
    seed: int = Field(0, ge=0, le=1_000_000)


class DeployReq(BaseModel):
    policy: str = Field("gated-v2field", max_length=40)
    profile: str = Field("warehouse", max_length=32)
    trials: int = Field(8, ge=2, le=MAX_DEPLOY_TRIALS)


class ProbeReq(BaseModel):
    policy: str = Field("gated-v2field", max_length=40)
    axis: str = Field("push_left", max_length=40)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "policies": len(POLICIES)}


@app.get("/api/policies")
def policies() -> dict:
    return {"policies": [{"name": k, "label": v["label"], "score": v["score"]} for k, v in POLICIES.items()]}


@app.get("/api/profiles")
def profiles() -> dict:
    return {"profiles": [{"name": p.name, "description": p.description} for p in PROFILES.values()]}


@app.get("/api/axes")
def axes() -> dict:
    return {"axes": [{"key": a.key, "label": a.label, "unit": a.unit} for a in AXIS_BY_KEY.values()]}


@app.post("/api/compile")
def api_compile(req: TextReq) -> dict:
    c = compile_text(req.text)
    return {"source": c.source, "matched": list(c.matched), "scenario": c.scenario.to_dict()}


@app.post("/api/compile_image")
def api_compile_image(req: ImageReq) -> dict:
    try:
        raw = base64.b64decode(req.image_b64.split(",")[-1], validate=True)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(400, "The image is not valid base64. Upload a JPEG or PNG file.") from e
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "The image is larger than 3 MB. Upload a smaller photo.")
    try:
        c = compile_image(raw, caption=req.caption)
    except Exception as e:  # noqa: BLE001 - decoder errors vary by format; report them as a client error
        raise HTTPException(400, f"Could not read the image ({str(e)[:80]}). Try a JPEG or PNG.") from e
    return {"source": c.source, "matched": list(c.matched), "scenario": c.scenario.to_dict()}


@app.post("/api/simulate")
def api_simulate(req: SimReq) -> dict:
    scn = _scenario(req.scenario)
    pol = _policy(req.policy)
    t0 = time.perf_counter()
    env = QuadrupedEnv(scn, seed=req.seed)
    out, frames = rollout(pol, scn, seed=req.seed, record=True, env=env)
    o = out.to_dict()
    return {"outcome": o, "explanation": explain(scn, o), "specs": env.geom_specs(),
            "frames": frames[::FRAME_STRIDE], "dt": CONTROL_DT * FRAME_STRIDE,
            "terrain": _terrain(scn, req.seed), "scenario": scn.to_dict(), "policy": req.policy,
            "compute_ms": round((time.perf_counter() - t0) * 1000)}


@app.post("/api/deploy")
def api_deploy(req: DeployReq) -> dict:
    if req.profile not in PROFILES:
        raise HTTPException(404, f"Unknown place '{req.profile[:32]}'. Choose one of: {', '.join(PROFILES)}.")
    _policy(req.policy)
    r = deployment_check(POLICIES[req.policy]["spec"], req.profile, trials=req.trials, workers=1)
    r["worst"] = [{"explanation": explain(Scenario.from_dict(w["scenario"]), w["outcome"])} for w in r["worst"]]
    return r


@app.post("/api/probe")
def api_probe(req: ProbeReq) -> dict:
    if req.axis not in AXIS_BY_KEY:
        raise HTTPException(404, f"Unknown stress '{req.axis[:40]}'. See /api/axes for the list.")
    _policy(req.policy)
    # same settings as the CLI reports: 6 s runs, 6 halvings, pushes at 4 gait phases
    return probe_axis((POLICIES[req.policy]["spec"], Scenario(duration_s=6.0).to_dict(), req.axis, [0], 6))
