"""Scenario schema: the single validated contract between language, search and physics.

Every scenario, whether typed by a person, produced by an LLM, inferred from a photo,
or proposed by the adversarial search, passes through ``Scenario.from_dict`` which
clamps every field to physically meaningful bounds. Nothing downstream ever sees an
unvalidated value.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

TERRAINS = ("flat", "rough", "slope", "stairs")

# (min, max, default) for every scalar field. Defaults describe a benign lab floor.
BOUNDS: dict[str, tuple[float, float, float]] = {
    "roughness": (0.0, 0.08, 0.0),  # metres of random height variation
    "slope_deg": (-20.0, 20.0, 0.0),  # positive = uphill in +x
    "step_height": (0.0, 0.10, 0.0),  # stair riser height in metres
    "friction": (0.05, 2.0, 1.0),
    "payload_kg": (0.0, 8.0, 0.0),
    "payload_offset_x": (-0.15, 0.15, 0.0),  # payload shift along body x (m)
    "motor_strength": (0.2, 1.5, 1.0),  # torque-limit multiplier
    "latency_steps": (0.0, 6.0, 0.0),  # control steps of action delay (20 ms each)
    "obs_noise": (0.0, 0.5, 0.0),  # std of gaussian noise on observations
    "target_speed": (0.0, 1.0, 0.5),  # commanded forward speed m/s
    "duration_s": (1.0, 30.0, 8.0),
}

PUSH_BOUNDS: dict[str, tuple[float, float, float]] = {
    "t": (0.0, 30.0, 2.0),  # seconds after start
    "force": (0.0, 400.0, 0.0),  # newtons, applied to the torso
    "direction_deg": (-180.0, 180.0, 90.0),  # 0 = from behind (pushes +x), 90 = pushes +y
    "duration": (0.02, 0.5, 0.1),  # seconds
}
MAX_PUSHES = 8


def _clamp(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return min(hi, max(lo, v))


@dataclass(frozen=True)
class Push:
    t: float = 2.0
    force: float = 0.0
    direction_deg: float = 90.0
    duration: float = 0.1

    @property
    def impulse(self) -> float:
        return self.force * self.duration

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Push:
        if not isinstance(d, dict):
            raise ValueError(f"push must be an object, got {type(d).__name__}")
        kw = {k: _clamp(d.get(k, dflt), lo, hi, dflt) for k, (lo, hi, dflt) in PUSH_BOUNDS.items()}
        return cls(**kw)


@dataclass(frozen=True)
class Scenario:
    name: str = "lab floor"
    terrain: str = "flat"
    roughness: float = 0.0
    slope_deg: float = 0.0
    step_height: float = 0.0
    friction: float = 1.0
    payload_kg: float = 0.0
    payload_offset_x: float = 0.0
    motor_strength: float = 1.0
    latency_steps: float = 0.0
    obs_noise: float = 0.0
    target_speed: float = 0.5
    duration_s: float = 8.0
    pushes: tuple[Push, ...] = field(default_factory=tuple)
    notes: str = ""

    # ------------------------------------------------------------------ io
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Scenario:
        """Validate and clamp an untrusted dict (LLM output, HTTP body, file)."""
        if not isinstance(d, dict):
            raise ValueError(f"scenario must be an object, got {type(d).__name__}")
        terrain = str(d.get("terrain", "flat")).lower().strip()
        if terrain not in TERRAINS:
            terrain = "flat"
        kw: dict[str, Any] = {
            k: _clamp(d.get(k, dflt), lo, hi, dflt) for k, (lo, hi, dflt) in BOUNDS.items()
        }
        kw["latency_steps"] = float(round(kw["latency_steps"]))
        raw_pushes = d.get("pushes") or []
        if not isinstance(raw_pushes, list):
            raise ValueError("pushes must be a list")
        pushes = tuple(Push.from_dict(p) for p in raw_pushes[:MAX_PUSHES])
        pushes = tuple(sorted((p for p in pushes if p.force > 0), key=lambda p: p.t))
        name = str(d.get("name", "scenario"))[:80] or "scenario"
        notes = str(d.get("notes", ""))[:500]
        return cls(name=name, terrain=terrain, pushes=pushes, notes=notes, **kw)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["pushes"] = [asdict(p) for p in self.pushes]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, text: str) -> Scenario:
        return cls.from_dict(json.loads(text))

    def with_(self, **changes: Any) -> Scenario:
        """Return a validated copy with ``changes`` applied (never mutates self)."""
        d = self.to_dict()
        d.update(changes)
        if "pushes" in changes:
            d["pushes"] = [
                p if isinstance(p, dict) else asdict(p) for p in changes["pushes"]
            ]
        return Scenario.from_dict(d)

    # ------------------------------------------------------------ analysis
    def severity(self) -> float:
        """A unitless 0..~1 'how hard is this world' score used to rank failures.

        Each stressor is normalised by its bound so a failure found at low severity
        is a more damning finding than one found at extreme settings.
        """
        terms = [
            self.roughness / BOUNDS["roughness"][1],
            abs(self.slope_deg) / BOUNDS["slope_deg"][1],
            self.step_height / BOUNDS["step_height"][1],
            max(0.0, 1.0 - self.friction) / (1.0 - BOUNDS["friction"][0]),
            self.payload_kg / BOUNDS["payload_kg"][1],
            max(0.0, 1.0 - self.motor_strength) / (1.0 - BOUNDS["motor_strength"][0]),
            self.latency_steps / BOUNDS["latency_steps"][1],
            self.obs_noise / BOUNDS["obs_noise"][1],
            max((p.impulse for p in self.pushes), default=0.0) / 60.0,
        ]
        return float(math.sqrt(sum(t * t for t in terms)))

    def dominant_stressor(self) -> str:
        scores = {
            "terrain": max(
                self.roughness / BOUNDS["roughness"][1],
                abs(self.slope_deg) / BOUNDS["slope_deg"][1],
                self.step_height / BOUNDS["step_height"][1],
            ),
            "low friction": max(0.0, 1.0 - self.friction) / 0.95,
            "payload": self.payload_kg / BOUNDS["payload_kg"][1],
            "weak motors": max(0.0, 1.0 - self.motor_strength) / 0.8,
            "latency": self.latency_steps / BOUNDS["latency_steps"][1],
            "sensor noise": self.obs_noise / BOUNDS["obs_noise"][1],
            "push": max((p.impulse for p in self.pushes), default=0.0) / 60.0,
        }
        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        return best if scores[best] > 0 else "none"


def json_schema() -> dict[str, Any]:
    """JSON schema handed to LLMs so they emit a structured scenario."""
    props: dict[str, Any] = {
        "name": {"type": "string"},
        "terrain": {"type": "string", "enum": list(TERRAINS)},
        "notes": {"type": "string"},
    }
    for k, (lo, hi, dflt) in BOUNDS.items():
        props[k] = {"type": "number", "minimum": lo, "maximum": hi, "default": dflt}
    props["pushes"] = {
        "type": "array",
        "maxItems": MAX_PUSHES,
        "items": {
            "type": "object",
            "properties": {
                k: {"type": "number", "minimum": lo, "maximum": hi, "default": dflt}
                for k, (lo, hi, dflt) in PUSH_BOUNDS.items()
            },
        },
    }
    return {"type": "object", "properties": props, "required": ["terrain"]}
