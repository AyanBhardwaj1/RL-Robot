"""Web playground: describe a world, watch the robot, shove it yourself."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import re
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .compiler import compile_image, compile_text
from .env import CONTROL_DT, QuadrupedEnv
from .evaluate import rollout
from .policy import load_policy
from .scenario import Push, Scenario
from .world import BASE_HEIGHT, HFIELD_HALF_X, HFIELD_HALF_Y, heightfield

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
STATIC = Path(__file__).resolve().parent / "static"
MAX_IMAGE_BYTES = 6_000_000
TERRAIN_GRID = 80
POLICY_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

app = FastAPI(title="Breakpoint")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def available_policies() -> dict[str, str | None]:
    out: dict[str, str | None] = {"cpg": None}
    for p in sorted(RUNS.glob("**/*.npz")):
        out[p.stem] = str(p)
    return out


def resolve_policy(name: str) -> str | None:
    if not POLICY_NAME.match(name or ""):
        raise HTTPException(400, "bad policy name")
    pol = available_policies()
    if name not in pol:
        raise HTTPException(404, f"unknown policy {name}")
    return pol[name]


def terrain_payload(scn: Scenario) -> dict:
    h = heightfield(scn) + BASE_HEIGHT
    idx_r = np.linspace(0, h.shape[0] - 1, TERRAIN_GRID).astype(int)
    idx_c = np.linspace(0, h.shape[1] - 1, TERRAIN_GRID * 2).astype(int)
    sub = h[np.ix_(idx_r, idx_c)]
    return {"half_x": HFIELD_HALF_X, "half_y": HFIELD_HALF_Y, "rows": len(idx_r), "cols": len(idx_c),
            "heights": [round(float(v), 4) for v in sub.ravel()], "friction": scn.friction}


class CompileReq(BaseModel):
    text: str = Field("", max_length=2000)


class ImageReq(BaseModel):
    image_b64: str = Field(..., max_length=MAX_IMAGE_BYTES * 2)
    caption: str = Field("", max_length=500)


class SimReq(BaseModel):
    scenario: dict
    policy: str = "cpg"
    seed: int = Field(0, ge=0, le=1_000_000)


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/policies")
def policies() -> dict:
    return {"policies": list(available_policies())}


@app.get("/api/reports")
def reports() -> dict:
    return {"reports": sorted(str(p.relative_to(RUNS).parent) for p in RUNS.glob("*/report.html"))}


@app.get("/reports/{name}")
def report(name: str) -> FileResponse:
    if not POLICY_NAME.match(name):
        raise HTTPException(400, "bad report name")
    path = (RUNS / name / "report.html").resolve()
    if RUNS.resolve() not in path.parents or not path.is_file():
        raise HTTPException(404, "no such report")
    return FileResponse(path)


@app.post("/api/compile")
def api_compile(req: CompileReq) -> dict:
    c = compile_text(req.text)
    return {"source": c.source, "matched": list(c.matched), "scenario": c.scenario.to_dict()}


@app.post("/api/compile_image")
def api_compile_image(req: ImageReq) -> dict:
    try:
        raw = base64.b64decode(req.image_b64.split(",")[-1], validate=True)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(400, "image_b64 is not valid base64") from e
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "image too large")
    try:
        c = compile_image(raw, caption=req.caption)
    except OSError as e:
        raise HTTPException(400, "could not decode image") from e
    return {"source": c.source, "matched": list(c.matched), "scenario": c.scenario.to_dict()}


@app.post("/api/simulate")
def api_simulate(req: SimReq) -> dict:
    try:
        scn = Scenario.from_dict(req.scenario)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    spec = resolve_policy(req.policy)
    env = QuadrupedEnv(scn, seed=req.seed)
    out, frames = rollout(load_policy(spec), scn, seed=req.seed, record=True, env=env)
    return {"outcome": out.to_dict(), "specs": env.geom_specs(), "frames": frames[::2], "dt": CONTROL_DT * 2,
            "terrain": terrain_payload(scn), "scenario": scn.to_dict()}


@app.websocket("/ws/play")
async def play(ws: WebSocket) -> None:
    """Interactive mode: the client sends shoves, the server streams poses at 25 Hz."""
    await ws.accept()
    try:
        init = json.loads(await ws.receive_text())
        scn = Scenario.from_dict(init.get("scenario") or {}).with_(duration_s=30.0, pushes=[])
        spec = resolve_policy(str(init.get("policy", "cpg")))
        policy = load_policy(spec)
        env = QuadrupedEnv(scn, seed=0)
        obs = env.reset(scn, seed=0)
        await ws.send_text(json.dumps({"type": "init", "specs": env.geom_specs(), "terrain": terrain_payload(scn)}))
        shoves, total = 0, 0.0
        tick = 0
        while True:
            try:
                msg = json.loads(await asyncio.wait_for(ws.receive_text(), timeout=0.001))
            except asyncio.TimeoutError:
                msg = None
            if msg and msg.get("type") == "shove":
                force = float(np.clip(float(msg.get("force", 80)), 0, 400))
                direction = float(np.clip(float(msg.get("direction_deg", 90)), -180, 180))
                p = Push(t=env.t + 0.001, force=force, direction_deg=direction, duration=0.1)
                env.scenario = env.scenario.with_(pushes=[*[q.__dict__ for q in env.scenario.pushes], p.__dict__][-8:])
                shoves += 1
                total += p.impulse
            obs, _, done, info = env.step(policy.act(obs))
            tick += 1
            if tick % 2 == 0:
                await ws.send_text(json.dumps({"type": "frame", "poses": env.body_poses(), "t": round(env.t, 2),
                                               "x": round(info.x, 2), "shoves": shoves, "impulse": round(total, 1)}))
                await asyncio.sleep(CONTROL_DT * 2)
            if info.fell or done:
                await ws.send_text(json.dumps({"type": "over", "fell": info.fell, "t": round(env.t, 2),
                                               "shoves": shoves, "impulse": round(total, 1), "x": round(info.x, 2)}))
                break
    except WebSocketDisconnect:
        return
    except HTTPException as e:
        await ws.send_text(json.dumps({"type": "error", "message": str(e.detail)[:200]}))
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        await ws.send_text(json.dumps({"type": "error", "message": str(e)[:200]}))
    await ws.close()
