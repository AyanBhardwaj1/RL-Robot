"""Language and image -> Scenario.

Design (from the research): the LLM never writes simulator code. It fills a bounded
JSON schema, and the result is clamped by ``Scenario.from_dict`` before use. A
deterministic rule parser handles the common phrasings offline and is the fallback
whenever no LLM is configured, the call fails, or the reply does not validate.

LLM backend: any OpenAI-compatible chat endpoint (Llama API, vLLM, Ollama, ...):
    BREAKPOINT_LLM_BASE_URL   e.g. https://api.llama.com/compat/v1
    BREAKPOINT_LLM_API_KEY
    BREAKPOINT_LLM_MODEL      e.g. a Llama 4 model id
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from io import BytesIO

import numpy as np

from .scenario import Scenario, json_schema

LLM_TIMEOUT_S = 20
NUM = r"(\d+(?:\.\d+)?)"


@dataclass(frozen=True)
class Compiled:
    scenario: Scenario
    source: str  # "rules" | "llm" | "image-heuristic" | "llm-vision"
    matched: tuple[str, ...]


# ----------------------------------------------------------------- rules
def _num_near(text: str, pattern: str, default: float) -> float:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else default


def compile_rules(text: str) -> Compiled:
    t = text.lower()
    d: dict = {"name": text.strip()[:60] or "scenario", "notes": text.strip()[:500]}
    hits: list[str] = []

    def hit(tag: str) -> None:
        hits.append(tag)

    # surfaces
    if re.search(r"\b(ice|icy|frozen|wet tile|polished|oil|oily|slippery)\b", t):
        d["friction"] = 0.12 if re.search(r"\b(ice|icy|oil|oily)\b", t) else 0.25
        hit("low friction")
    elif re.search(r"\b(mud|muddy|wet|rain|sand|sandy|snow)\b", t):
        d["friction"] = 0.4
        hit("reduced friction")
    if m := re.search(r"friction (?:of |=|is )?" + NUM, t):
        d["friction"] = float(m.group(1))
        hit("explicit friction")

    # terrain
    if re.search(r"\b(stairs?|steps?|staircase|curb)\b", t):
        d["terrain"] = "stairs"
        cm = _num_near(t, NUM + r"\s*cm", 6.0)
        d["step_height"] = cm / 100.0
        hit("stairs")
    elif re.search(r"\b(hill|slope|ramp|incline|uphill|downhill|decline)\b", t):
        d["terrain"] = "slope"
        deg = _num_near(t, NUM + r"\s*(?:deg|degree|°)", 10.0)
        d["slope_deg"] = -deg if re.search(r"\b(downhill|decline|down)\b", t) else deg
        hit("slope")
    if re.search(r"\b(rough|rocky|rubble|gravel|uneven|bumpy|forest|trail|grass)\b", t):
        d["terrain"] = d.get("terrain", "rough") if d.get("terrain") in ("slope", "stairs") else "rough"
        d["roughness"] = 0.05 if re.search(r"\b(rubble|rocky|very)\b", t) else 0.03
        hit("rough terrain")

    # payload
    if m := re.search(NUM + r"\s*(?:kg|kilo)", t):
        d["payload_kg"] = float(m.group(1))
        hit("payload")
    elif re.search(r"\b(carry|carrying|backpack|payload|box|package|load)\b", t):
        d["payload_kg"] = 2.5
        hit("payload")
    if re.search(r"\b(front[- ]heavy|leaning forward)\b", t):
        d["payload_offset_x"] = 0.1
    elif re.search(r"\b(back[- ]heavy|rear[- ]heavy)\b", t):
        d["payload_offset_x"] = -0.1

    # actuation / sensing
    if re.search(r"\b(weak|worn|tired|damaged|old|low battery|overheat\w*)\b", t):
        pct = _num_near(t, NUM + r"\s*%", 60.0)
        d["motor_strength"] = pct / 100.0 if pct <= 100 else 0.6
        hit("weak motors")
    if re.search(r"\b(lag|laggy|latency|delay|delayed|slow network|wifi)\b", t):
        ms = _num_near(t, NUM + r"\s*ms", 60.0)
        d["latency_steps"] = round(ms / 20.0)
        hit("latency")
    if re.search(r"\b(noisy|noise|foggy|glitch\w*|faulty sensor|vibration)\b", t):
        d["obs_noise"] = 0.15
        hit("sensor noise")

    # speed
    if re.search(r"\b(run|running|sprint|fast|hurry)\b", t):
        d["target_speed"] = 0.8
        hit("fast")
    elif re.search(r"\b(slow|careful|creep)\b", t):
        d["target_speed"] = 0.25

    # pushes: "kicked from the left at 3s with 60 N", "shoved", "wind gust", "bumped"
    pushes = []
    for m in re.finditer(r"\b(kick\w*|push\w*|shove\w*|bump\w*|hit|gust\w*|knock\w*|collid\w*)\b", t):
        window = t[m.start(): m.start() + 80]
        direction = 90.0
        if re.search(r"\bright\b", window):
            direction = -90.0
        elif re.search(r"\b(front|ahead|face)\b", window):
            direction = 180.0
        elif re.search(r"\b(behind|back|rear)\b", window):
            direction = 0.0
        force = _num_near(window, NUM + r"\s*n\b", 0.0)
        if not force:
            force = 150.0 if re.search(r"\b(hard|strong|violent\w*|big)\b", window) else 80.0
        when = _num_near(window, r"(?:at|after)\s*" + NUM + r"\s*s", 2.0 + 1.5 * len(pushes))
        pushes.append({"t": when, "force": force, "direction_deg": direction, "duration": 0.1})
    if re.search(r"\b(wind|windy|storm)\b", t) and not pushes:
        pushes = [{"t": 2.0 + i * 1.5, "force": 40.0, "direction_deg": 90.0, "duration": 0.4} for i in range(3)]
    if pushes:
        d["pushes"] = pushes
        hit("pushes")
    if "terrain" not in d:
        d["terrain"] = "flat"
    return Compiled(Scenario.from_dict(d), "rules", tuple(hits))


# ------------------------------------------------------------------- LLM
SYSTEM_PROMPT = (
    "You convert descriptions of real-world conditions into a physics stress-test scenario "
    "for a 6 kg quadruped robot. Reply with ONLY a JSON object matching this JSON schema. "
    "Use physically realistic values: ice friction ~0.1, wet tile ~0.3, concrete ~1.0; a firm "
    "human shove is 60-120 N for 0.1 s; direction_deg 0 pushes the robot forward from behind, "
    "90 pushes it toward its left, -90 toward its right. Schema:\n"
)


def _llm_config() -> tuple[str, str, str] | None:
    base = os.environ.get("BREAKPOINT_LLM_BASE_URL", "").rstrip("/")
    key = os.environ.get("BREAKPOINT_LLM_API_KEY", "")
    model = os.environ.get("BREAKPOINT_LLM_MODEL", "")
    if not (base and key and model):
        return None
    if not base.startswith("https://") and not base.startswith("http://localhost"):
        raise ValueError("BREAKPOINT_LLM_BASE_URL must be https (or http://localhost)")
    return base, key, model


def _chat(messages: list[dict]) -> str:
    cfg = _llm_config()
    if cfg is None:
        raise RuntimeError("no LLM configured")
    base, key, model = cfg
    body = json.dumps({"model": model, "messages": messages, "temperature": 0.2,
                       "response_format": {"type": "json_object"}}).encode()
    req = urllib.request.Request(f"{base}/chat/completions", data=body, method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT_S) as resp:
        payload = json.loads(resp.read().decode())
    return payload["choices"][0]["message"]["content"]


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("model reply contained no JSON object")
    return json.loads(m.group(0))


def compile_text(text: str, use_llm: bool = True) -> Compiled:
    """Natural language -> validated Scenario. Falls back to rules on any LLM problem."""
    text = (text or "").strip()[:2000]
    if not text:
        return Compiled(Scenario(), "rules", ())
    if use_llm and _llm_config() is not None:
        try:
            reply = _chat([
                {"role": "system", "content": SYSTEM_PROMPT + json.dumps(json_schema())},
                {"role": "user", "content": text},
            ])
            scn = Scenario.from_dict(_parse_json(reply))
            return Compiled(scn.with_(notes=text[:500]), "llm", ("llm",))
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, RuntimeError, json.JSONDecodeError):
            pass  # deterministic fallback below; rules never fail
    return compile_rules(text)


# ----------------------------------------------------------------- image
def image_features(img_bytes: bytes) -> dict[str, float]:
    """Cheap, explainable surface cues from a photo of the ground."""
    from PIL import Image

    im = Image.open(BytesIO(img_bytes)).convert("RGB")
    im.thumbnail((256, 256))
    a = np.asarray(im, dtype=np.float64) / 255.0
    gray = a.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    r, g, b = a[..., 0].mean(), a[..., 1].mean(), a[..., 2].mean()
    sat = float((a.max(axis=2) - a.min(axis=2)).mean())
    # Specular highlights (near-white, low saturation pixels) suggest wet/icy/polished floors
    spec = float(((gray > 0.85) & ((a.max(axis=2) - a.min(axis=2)) < 0.08)).mean())
    rows = gray.mean(axis=1)
    vertical_bands = float(np.abs(np.diff(rows)).max()) if len(rows) > 1 else 0.0
    return {"texture": float(gx + gy), "brightness": float(gray.mean()), "red": float(r), "green": float(g),
            "blue": float(b), "saturation": sat, "specular": spec, "banding": vertical_bands}


def compile_image(img_bytes: bytes, caption: str = "", use_llm: bool = True) -> Compiled:
    if use_llm and _llm_config() is not None:
        try:
            b64 = base64.b64encode(img_bytes).decode()
            reply = _chat([
                {"role": "system", "content": SYSTEM_PROMPT + json.dumps(json_schema())},
                {"role": "user", "content": [
                    {"type": "text", "text": "Build a stress test of the ground in this photo. " + caption},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]},
            ])
            return Compiled(Scenario.from_dict(_parse_json(reply)), "llm-vision", ("llm-vision",))
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, RuntimeError, json.JSONDecodeError):
            pass
    f = image_features(img_bytes)
    d: dict = {"name": "from photo" + (f": {caption[:40]}" if caption else ""), "terrain": "flat",
               "notes": "image heuristic: " + ", ".join(f"{k}={v:.3f}" for k, v in f.items())}
    hits = []
    if f["texture"] > 0.12:
        d["terrain"], d["roughness"] = "rough", float(np.clip((f["texture"] - 0.08) * 0.5, 0.01, 0.07))
        hits.append("high texture -> rough")
    if f["specular"] > 0.04 or (f["blue"] > f["red"] + 0.05 and f["brightness"] > 0.6):
        d["friction"] = 0.15
        hits.append("glare/cold tones -> low friction")
    elif f["green"] > max(f["red"], f["blue"]) + 0.04:
        d["friction"], d["roughness"] = 0.6, max(d.get("roughness", 0.0), 0.02)
        d["terrain"] = "rough"
        hits.append("vegetation -> grass")
    elif f["red"] > f["blue"] + 0.08 and f["saturation"] > 0.15:
        d["friction"] = 0.45
        hits.append("warm/brown -> dirt or mud")
    if f["banding"] > 0.15:
        d["terrain"], d["step_height"] = "stairs", 0.05
        hits.append("strong horizontal bands -> stairs")
    base = Scenario.from_dict(d)
    if caption:
        extra = compile_rules(caption)
        merged = {**base.to_dict(), **{k: v for k, v in extra.scenario.to_dict().items()
                                       if v != Scenario().to_dict().get(k) and k not in ("name", "notes")}}
        base = Scenario.from_dict(merged)
        hits.extend(extra.matched)
    return Compiled(base, "image-heuristic", tuple(hits))
