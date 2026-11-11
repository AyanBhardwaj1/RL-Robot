"""From measured failures to a hazard log.

A hazard analysis usually starts from a workshop and expert estimates of how often things happen. Here the
frequencies come from simulation: every failure the humanoid lab records is classified into a hazard, counted
per job and body (biped or wheeled), given a 95% upper bound, and paired with a mitigation and the exact
command that re-verifies it. Severity is assigned per hazard class (1 = negligible … 4 = severe) the way a
safety engineer would before review; the point is that the *probability* column is now evidence, not opinion.
"""

from __future__ import annotations

from dataclasses import dataclass

from .evaluate import failure_upper_bound
from .humanoid import JOBS, Humanoid, WheeledBase, job_risk


@dataclass(frozen=True)
class HazardClass:
    hid: str
    title: str
    cause_prefix: str  # matches the start of a recovery failure reason
    harm: str
    severity: int  # 1 negligible, 2 minor, 3 serious, 4 severe
    mitigation: str
    verify: str


HAZARDS = (
    HazardClass("H1", "Fall after a shove the robot cannot step out of", "needed more than",
                "Robot falls onto a person, a product or itself", 3,
                "Speed and separation monitoring near people; controlled-fall behaviour; widen step reach and swing speed",
                "python -m breakpoint humanoid  # push envelope per direction"),
    HazardClass("H2", "Loss of footing on a slippery floor", "slipped",
                "Uncontrolled slide or fall during a carry", 3,
                "Floor-condition detection and a friction-aware gait; spill response procedure",
                "python -m breakpoint humanoid --friction 0.3"),
    HazardClass("H3", "Unsettled recovery: still swaying at the end of the window", "still moving",
                "Delayed task, secondary contact with surroundings", 2,
                "Damping in the balance controller; pause task until settled",
                "python -m breakpoint humanoid --latency-ms 150"),
    HazardClass("H4", "Route blocked by a curb, threshold or stair", "blocked",
                "Mission failure; tipping if the base tries to climb", 2,
                "Route survey and ramps, or a legged platform where steps exist",
                "python -m breakpoint humanoid  # job table: trailer unloading, home care"),
    HazardClass("H5", "Wheeled base tips over: it cannot drive under the capture point", "tipped over",
                "Tip-over onto a person or rack", 4,
                "Lower payload height, acceleration headroom, speed limits near people",
                "python -m breakpoint humanoid --payload 25"),
    HazardClass("H6", "Capture point runs away beyond leg reach", "capture point ran away",
                "Fall onto a person, a product or itself", 3,
                "Same as H1; detect early and trigger a protective stop",
                "python -m breakpoint humanoid"),
)

RISK = {  # severity x likelihood band -> risk level, a common 4x4 matrix
    1: ("low", "low", "low", "medium"),
    2: ("low", "low", "medium", "high"),
    3: ("low", "medium", "high", "high"),
    4: ("medium", "high", "high", "high"),
}


def likelihood_band(rate_upper: float) -> int:
    """0 improbable (<1%), 1 remote (<5%), 2 occasional (<20%), 3 frequent — judged on the 95% upper bound."""
    if rate_upper < 0.01:
        return 0
    if rate_upper < 0.05:
        return 1
    if rate_upper < 0.20:
        return 2
    return 3


def classify(reason: str) -> HazardClass | None:
    for h in HAZARDS:
        if reason.startswith(h.cause_prefix):
            return h
    return None


def hazard_log(trials: int = 150, seed: int = 0) -> list[dict]:
    rows = []
    bodies = {"biped": Humanoid(), "wheeled": WheeledBase()}
    for job in JOBS:
        for body_name, body in bodies.items():
            r = job_risk(job, body, trials=trials, seed=seed)
            for cause, n in r["causes"].items():
                h = classify(cause)
                if h is None:
                    continue
                ub = failure_upper_bound(trials, n)
                band = likelihood_band(ub)
                rows.append({"hazard": h.hid, "title": h.title, "job": job, "body": body_name, "events": n,
                             "trials": trials, "rate": n / trials, "rate_upper_95": round(ub, 4),
                             "severity": h.severity, "likelihood": ["improbable", "remote", "occasional", "frequent"][band],
                             "risk": RISK[h.severity][band], "harm": h.harm, "mitigation": h.mitigation, "verify": h.verify})
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(rows, key=lambda r: (order[r["risk"]], -r["rate_upper_95"]))


def to_markdown(rows: list[dict], trials: int) -> str:
    lines = ["# Hazard log (generated)", "",
             (f"Frequencies are measured: {trials} sampled shifts per job and body in the humanoid lab, with a 95% upper "
              "bound. Severity is assigned per hazard class and must be reviewed by a safety engineer. Hazards with "
              "no recorded events are not listed; that is evidence of rarity within the budget, not proof of absence."), "",
             "| Risk | ID | Hazard | Job | Body | Events | Rate (≤ 95%) | Severity | Mitigation | Verify |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['risk']} | {r['hazard']} | {r['title']} | {r['job']} | {r['body']} | {r['events']}/{r['trials']} | "
                     f"{r['rate']:.1%} (≤ {r['rate_upper_95']:.1%}) | {r['severity']} | {r['mitigation']} | `{r['verify']}` |")
    if not rows:
        lines.append("| – | – | No failures recorded | – | – | – | – | – | – | – |")
    return "\n".join(lines) + "\n"
