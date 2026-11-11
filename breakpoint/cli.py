"""breakpoint CLI.

    breakpoint compile "icy loading dock, carrying 3kg, shoved from the left"
    breakpoint run --policy runs/policy_v1.npz --scenario scn.json
    breakpoint train --steps 200000 --out runs/policy.npz
    breakpoint audit --policy runs/policy_v1.npz --out runs/audit
    breakpoint harden --policy runs/policy_v1.npz --out runs/hardened
    breakpoint regress --policy runs/policy_v2.npz --suite suites/
    breakpoint serve
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .compiler import compile_image, compile_text
from .evaluate import evaluate_many, rollout
from .hunt import hunt, shrink
from .policy import ResidualPolicy, load_policy
from .ppo import EVAL_WORLDS, PPOConfig, ScenarioSampler, deterministic_score, train
from .probe import probe
from .profiles import PROFILES, deployment_check
from .redteam import boundary_map, compound_pairs, interactions, red_team
from .report import render_html, scorecard
from .scenario import Scenario

NOMINAL_TRIALS = 24


def _log(row: dict) -> None:
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)


def nominal_trials(spec: str | None, n: int = NOMINAL_TRIALS, seed: int = 123) -> list[dict]:
    sampler = ScenarioSampler(seed=seed, widen_frac=0.0)
    scns = [sampler.sample(6.0) for _ in range(n)]
    return [r[0] for r in evaluate_many(spec, scns, seeds=[seed])]


DEPLOY_TRIALS = 30


def audit(spec: str | None, label: str, hunt_gens: int = 5, workers: int = 3,
          deploy_trials: int = DEPLOY_TRIALS) -> dict:
    print(f"[audit] {label}: probing breaking points", flush=True)
    pr = probe(spec, workers=workers)
    print(f"[audit] {label}: nominal trials", flush=True)
    nom = nominal_trials(spec)
    print(f"[audit] {label}: adversarial hunt", flush=True)
    h = hunt(spec, generations=hunt_gens, workers=workers, log=_log)
    card = scorecard(pr, nom, h.atlas)
    print(f"[audit] {label}: compound stress + red team", flush=True)
    extras = {
        "interactions": interactions(spec, pr, workers=workers),
        "boundary": boundary_map(spec, "friction", _weakest_push(pr), n=6, workers=workers),
        "redteam": red_team(spec, pr, workers=workers),
    }
    card["compound_failures"] = compound_pairs(extras["interactions"])
    card["redteam_failures"] = sum(a["pass_rate"] < 1 for a in extras["redteam"]["attacks"])
    deploy = {}
    if deploy_trials > 0:
        print(f"[audit] {label}: deployment profiles", flush=True)
        for name in PROFILES:
            r = deployment_check(spec, name, trials=deploy_trials, workers=workers)
            deploy[name] = {k: r[k] for k in ("trials", "failures", "failure_rate", "failure_rate_upper_95",
                                              "failures_by_dominant_stressor")}
    return {"label": label, "policy": spec or "cpg", "probe": pr, "nominal": nom, "hunt": h.to_dict(),
            "card": card, "extras": extras, "deploy": deploy}


def _weakest_push(pr: list[dict]) -> str:
    pushes = [r for r in pr if r["key"].startswith("push_")]
    return min(pushes, key=lambda r: r["fraction"])["key"] if pushes else "push_left"


def _write_report(out: Path, title: str, audits: list[dict], notes: list[str]) -> Path:
    last = audits[-1]
    page = render_html(title, {a["label"]: a["card"] for a in audits}, {a["label"]: a["probe"] for a in audits},
                       last["hunt"]["atlas"], last["hunt"], notes, last.get("extras"),
                       deploy={a["label"]: a["deploy"] for a in audits if a.get("deploy")})
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.html").write_text(page)
    (out / "audit.json").write_text(json.dumps(audits, indent=1))
    return out / "report.html"


def save_suite(atlas: dict, spec: str | None, suite_dir: Path, shrink_iters: int = 5) -> list[Path]:
    """Shrink every atlas failure to a minimal counterexample and save it as a regression test."""
    suite_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, cell in enumerate(sorted(atlas.values(), key=lambda c: c["severity"])):
        scn = Scenario.from_dict(cell["scenario"])
        minimal, alpha = shrink(spec, scn, seed=cell["seed"], iters=shrink_iters)
        case = {"scenario": minimal.to_dict(), "seed": cell["seed"], "alpha": alpha,
                "stressor": cell["stressor"], "found_against": spec or "cpg"}
        p = suite_dir / f"case_{i:02d}_{cell['stressor'].replace(' ', '_')}.json"
        p.write_text(json.dumps(case, indent=1))
        paths.append(p)
    return paths


def cmd_compile(a: argparse.Namespace) -> int:
    if a.image:
        c = compile_image(Path(a.image).read_bytes(), caption=a.text or "", use_llm=not a.offline)
    else:
        c = compile_text(a.text, use_llm=not a.offline)
    print(json.dumps({"source": c.source, "matched": list(c.matched), "scenario": c.scenario.to_dict()}, indent=2))
    if a.out:
        Path(a.out).write_text(c.scenario.to_json())
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    scn = Scenario.from_json(Path(a.scenario).read_text()) if a.scenario else compile_text(a.text or "").scenario
    out, _ = rollout(load_policy(a.policy), scn, seed=a.seed)
    print(json.dumps(out.to_dict(), indent=2))
    return 0 if out.passed else 1


def cmd_train(a: argparse.Namespace) -> int:
    pol = ResidualPolicy.load(a.init) if a.init else ResidualPolicy(seed=a.seed, name=Path(a.out).stem)
    failures = [Scenario.from_dict(json.loads(p.read_text())["scenario"]) for p in sorted(Path(a.suite).glob("*.json"))] if a.suite else []
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    best = {"score": -1.0, "steps": 0}

    def keep_best(policy: ResidualPolicy, row: dict) -> None:
        # model selection on the deterministic policy: the stochastic training curve can lie
        if row["det_score"] > best["score"]:
            best.update(score=row["det_score"], steps=row["steps"])
            policy.save(a.out)

    train(pol, ScenarioSampler(failures=failures, seed=a.seed), PPOConfig(total_steps=a.steps), seed=a.seed,
          log=_log, checkpoint=keep_best)
    print(f"saved {a.out} (best deterministic score {best['score']:.3f} at {best['steps']} steps)")
    return 0


def cmd_audit(a: argparse.Namespace) -> int:
    audits = []
    if a.baseline:
        audits.append(audit(None, "cpg-baseline", a.hunt_gens, a.workers))
    audits.append(audit(a.policy, Path(a.policy).stem if a.policy else "cpg-baseline", a.hunt_gens, a.workers))
    path = _write_report(Path(a.out), a.title, audits, [])
    print(json.dumps(audits[-1]["card"], indent=2))
    print(f"report: {path}")
    return 0


def policy_label(spec: str | None) -> str:
    """Human label for a policy spec: cpg-baseline, policy_v2, gated-policy_v2, my_controller."""
    if spec is None or spec == "cpg":
        return "cpg-baseline"
    if spec.startswith("gated:"):
        return "gated-" + policy_label(spec[6:])
    if spec.startswith("py:"):
        return spec.rsplit(":", 1)[-1]
    return Path(spec).stem


def cmd_leaderboard(a: argparse.Namespace) -> int:
    if len(a.policies) > 4:
        print(f"warning: only the first 4 of {len(a.policies)} policies fit in one report", file=sys.stderr)
    specs = list(dict.fromkeys(None if p == "cpg" else p for p in a.policies))[:4]  # de-duplicate, keep order
    for sp in specs:
        load_policy(sp)  # fail fast on a bad spec, before minutes of auditing
    labels: list[str] = []
    for sp in specs:
        lab = policy_label(sp)
        labels.append(lab if lab not in labels else f"{Path(sp.split(':')[-1]).parent.name}/{lab}")
    audits = [audit(sp, lab, a.hunt_gens, a.workers) for sp, lab in zip(specs, labels)]
    notes = ["All policies audited in one run with identical code, seeds and search budget."]
    path = _write_report(Path(a.out), a.title, audits, notes)
    for x in audits:
        c = x["card"]
        print(f"{x['label']:>20}  {c['score']:5.1f}  {c['grade']:<11}  margin {c['margin_index']:.2f}  "
              f"pass {c['nominal_pass_rate']:.0%}  compound {c['compound_failures']}  redteam {c['redteam_failures']}")
    print(f"report: {path}")
    return 0


def cmd_harden(a: argparse.Namespace) -> int:
    out = Path(a.out)
    # validate inputs up front: the "before" audit takes minutes
    if a.policy and not a.policy.endswith(".npz"):
        raise SystemExit("harden needs a residual PPO policy (.npz) to fine-tune; 'cpg' and py: specs can only be audited")
    unknown = [n for n in (x.strip() for x in (a.profile or "").split(",")) if n and n not in PROFILES]
    if unknown:
        raise SystemExit(f"unknown profile(s) {unknown}; choose from {sorted(PROFILES)}")
    if a.policy:
        ResidualPolicy.load(a.policy)
    before = audit(a.policy, "before", a.hunt_gens, a.workers)
    suite = save_suite(before["hunt"]["atlas"], a.policy, out / "suite")
    print(f"[harden] {len(suite)} minimal counterexamples saved to {out / 'suite'}", flush=True)
    failures = [Scenario.from_dict(json.loads(p.read_text())["scenario"]) for p in suite]
    pol = ResidualPolicy.load(a.policy) if a.policy else ResidualPolicy(seed=a.seed)
    pol.name = "hardened"
    hardened = out / f"policy_{out.name}.npz"
    profiles = [PROFILES[n.strip()] for n in (a.profile or "").split(",") if n.strip()]
    # Select checkpoints on what hardening is *for*: generic worlds + the counterexamples
    # + (if given) worlds drawn from the target deployment profiles.
    sel_rng = np.random.default_rng(a.seed + 99)
    selection = [*EVAL_WORLDS, *failures[:8], *[p.sample(sel_rng) for p in profiles for _ in range(6)]]

    def score(policy: ResidualPolicy) -> float:
        return deterministic_score(policy, seeds=(0,), worlds=selection)

    start = score(pol)
    best = {"score": start}
    pol.save(hardened)  # the starting point is a valid candidate: never ship something worse

    def keep_best(policy: ResidualPolicy, row: dict) -> None:
        sc = score(policy)
        row["selection_score"] = round(sc, 4)
        if sc > best["score"]:
            best["score"] = sc
            policy.save(hardened)

    sampler = ScenarioSampler(failures=failures, seed=a.seed + 1, profiles=profiles,
                              widen_frac=0.35 if profiles else 0.2)
    train(pol, sampler, PPOConfig(total_steps=a.steps, lr=2e-4),
          seed=a.seed + 1, log=_log, checkpoint=keep_best)
    print(f"[harden] deterministic score {start:.3f} -> best {best['score']:.3f}", flush=True)
    after = audit(str(hardened), "after", a.hunt_gens, a.workers)
    reg = evaluate_many(str(hardened), failures, seeds=[0, 1])
    fixed = sum(all(r["passed"] for r in rs) for rs in reg)
    notes = [(f"Hardening replayed {len(failures)} shrunk counterexamples (30% of episodes) for {a.steps} PPO steps; "
              f"{fixed}/{len(failures)} now pass on both seeds.")]
    path = _write_report(out, a.title, [before, after], notes)
    print(notes[0])
    print(f"report: {path}")
    return 0


def cmd_regress(a: argparse.Namespace) -> int:
    cases = sorted(Path(a.suite).glob("*.json"))
    if not cases:
        print("no cases found", file=sys.stderr)
        return 2
    scns = [Scenario.from_dict(json.loads(p.read_text())["scenario"]) for p in cases]
    res = evaluate_many(a.policy, scns, seeds=[0, 1, 2])
    failed = 0
    for p, rs in zip(cases, res):
        ok = all(r["passed"] for r in rs)
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {p.name}  ({sum(r['passed'] for r in rs)}/3 seeds)")
    print(f"{len(cases) - failed}/{len(cases)} regression cases pass")
    return 1 if failed else 0


def cmd_redteam(a: argparse.Namespace) -> int:
    pr = probe(a.policy, workers=a.workers)
    res = red_team(a.policy, pr, n=a.n, workers=a.workers, use_llm=not a.offline)
    ix = interactions(a.policy, pr, workers=a.workers)
    print(json.dumps({"source": res["source"],
                      "attacks": [{"name": x["scenario"]["name"], "pass_rate": x["pass_rate"]} for x in res["attacks"]],
                      "compound_failures": [f"{c['a']} + {c['b']} @ {c['fraction']:.0%}" for c in ix if c["compound_failure"]]},
                     indent=2))
    return 0


def cmd_deploy(a: argparse.Namespace) -> int:
    names = sorted(PROFILES) if a.profile == "all" else [a.profile]
    rows = []
    for name in names:
        r = deployment_check(a.policy, name, trials=a.trials, seed=a.seed, workers=a.workers)
        rows.append(r)
        print(f"{name:>13}: {r['failures']}/{r['trials']} fail  ({r['failure_rate']:.1%}, 95% UB {r['failure_rate_upper_95']:.1%})"
              f"  causes {r['failures_by_dominant_stressor']}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))
    return 0 if all(r["failure_rate_upper_95"] <= a.max_fail for r in rows) else 1


def _sha256(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    import subprocess

    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def certificate(spec: str | None, result: dict) -> dict:
    """Traceable readiness certificate: what was tested, with what, and what it scored."""
    import platform
    from datetime import datetime, timezone

    import mujoco

    from . import __version__

    file_part = spec.split(":")[-1] if spec else None
    return {
        "policy": policy_label(spec), "policy_spec": spec or "cpg",
        "policy_sha256": _sha256(file_part) if file_part and file_part.endswith(".npz") else None,
        "issued_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "toolchain": {"breakpoint": __version__, "mujoco": mujoco.__version__, "numpy": np.__version__,
                      "python": platform.python_version(), "git_commit": _git_commit()},
        "card": result["card"],
        "margins": {r["key"]: {"breaking_point": r["breaking_point"], "unit": r["unit"]} for r in result["probe"]},
        "deployment": result.get("deploy", {}),
        "search_budget": {"adversarial_rollouts": result["hunt"]["evaluations"],
                          "nominal_trials": result["card"]["nominal_trials"]},
    }


def cmd_certify(a: argparse.Namespace) -> int:
    spec = None if a.policy in (None, "cpg") else a.policy
    load_policy(spec)  # fail fast
    result = audit(spec, policy_label(spec), a.hunt_gens, a.workers, deploy_trials=a.deploy_trials)
    cert = certificate(spec, result)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(cert, indent=1))
    ok = cert["card"]["score"] >= a.min_score
    print(f"{cert['policy']}: {cert['card']['score']} {cert['card']['grade']} -> "
          f"{'PASS' if ok else 'FAIL'} (min {a.min_score}); certificate: {a.out}")
    return 0 if ok else 1


def cmd_humanoid(a: argparse.Namespace) -> int:
    from .humanoid import JOBS, Conditions, Humanoid, WheeledBase, envelope, job_risk

    cond = Conditions(payload_kg=a.payload, friction=a.friction, latency_s=a.latency_ms / 1000.0)
    bodies = {"biped": Humanoid(), "wheeled": WheeledBase()}
    out = {"conditions": cond.to_dict(), "envelope": {}, "jobs": {}}
    print(f"Push-recovery envelope (largest survivable shove, N·s) · payload {a.payload} kg, μ {a.friction}, "
          f"latency {a.latency_ms:.0f} ms")
    for name, body in bodies.items():
        env = envelope(body, cond)
        out["envelope"][name] = env
        print(f"  {name:>8}: " + "  ".join(f"{k} {v:.0f}" for k, v in env.items()))
    print(f"\nFailure odds by job ({a.trials} sampled shifts each, 95% upper bound in brackets)")
    for job in JOBS:
        row = {n: job_risk(job, b, trials=a.trials, seed=a.seed) for n, b in bodies.items()}
        out["jobs"][job] = row
        cells = "   ".join(f"{n} {r['failure_rate']:.1%} (≤{r['failure_rate_upper_95']:.1%})" for n, r in row.items())
        print(f"  {job:<30} {cells}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {a.out}")
    return 0


def cmd_quantize(a: argparse.Namespace) -> int:
    from .quantize import sample_observations, save_compressed, stats

    base = ResidualPolicy.load(a.policy)
    obs = sample_observations()
    out_dir = Path(a.out)
    rows = []
    print(f"{'mode':<6}{'weights':>10}{'mean |Δa|':>12}{'max |Δa|':>11}{'µs/act':>9}{'score':>8}{'grade':>12}{'teleop fail':>13}")
    for mode in ("fp32", "fp16", "int8"):
        st = stats(base, mode, obs)
        path = save_compressed(a.policy, mode, out_dir)
        spec = f"gated:{path}" if a.gated else str(path)
        pr = probe(spec, workers=a.workers)
        nom = nominal_trials(spec)
        card = scorecard(pr, nom, {})
        dep = deployment_check(spec, "teleop", trials=a.trials, workers=a.workers)
        row = {**st.__dict__, "score": card["score"], "grade": card["grade"], "margin_index": card["margin_index"],
               "teleop_failure_rate": dep["failure_rate"], "breaking_points": {r["key"]: r["breaking_point"] for r in pr}}
        rows.append(row)
        print(f"{mode:<6}{st.weight_bytes:>10}{st.mean_abs_action_error:>12.5f}{st.max_abs_action_error:>11.5f}"
              f"{st.act_us:>9.1f}{card['score']:>8.1f}{card['grade']:>12}{dep['failure_rate']:>13.1%}")
    (out_dir / "quantize.json").write_text(json.dumps(rows, indent=1))
    print(f"wrote {out_dir / 'quantize.json'}")
    return 0


def cmd_design(a: argparse.Namespace) -> int:
    from .design import lever_table, to_text

    table = lever_table(trials=a.trials, seed=a.seed)
    print(to_text(table))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(table, indent=1))
        print(f"\nwrote {a.out}")
    return 0


def cmd_hazards(a: argparse.Namespace) -> int:
    from .hazards import hazard_log, to_markdown

    rows = hazard_log(trials=a.trials, seed=a.seed)
    md = to_markdown(rows, a.trials)
    print(md)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(md)
        Path(a.out).with_suffix(".json").write_text(json.dumps(rows, indent=1))
    return 1 if any(r["risk"] == "high" for r in rows) and a.fail_on_high else 0


def cmd_fleet(a: argparse.Namespace) -> int:
    from .fleet import read_log, site_odds, synthetic_log

    if a.log:
        records, name = read_log(a.log), Path(a.log).stem
    else:
        records, name = synthetic_log(a.demo, shifts=a.shifts, seed=a.seed), f"demo: {a.demo}"
        if a.write_demo:
            Path(a.write_demo).parent.mkdir(parents=True, exist_ok=True)
            Path(a.write_demo).write_text("\n".join(json.dumps(r) for r in records) + "\n")
    res = site_odds(records, name, trials=a.trials, seed=a.seed)
    p = res["profile"]
    print(f"Fitted profile from {len(records)} shifts ({name})")
    for k in ("payload", "friction", "latency", "slope", "push_impulse", "floor_step"):
        print(f"  {k:<13} {p[k][0]:g} … {p[k][1]:g}")
    print(f"  contact rate  {p['push_prob']:.0%} of shifts · steps on {p['step_prob']:.0%} of routes")
    for body, r in res["odds"].items():
        print(f"  {body:<8} fails {r['failure_rate']:.1%} (≤ {r['failure_rate_upper_95']:.1%})  causes {r['causes']}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("breakpoint.server:app", host=a.host, port=a.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="breakpoint", description="Find where your legged robot breaks before reality does.")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compile", help="language/photo -> scenario JSON")
    c.add_argument("text", nargs="?", default="")
    c.add_argument("--image")
    c.add_argument("--offline", action="store_true", help="never call an LLM")
    c.add_argument("--out")
    c.set_defaults(fn=cmd_compile)

    r = sub.add_parser("run", help="roll out one scenario")
    r.add_argument("--policy")
    r.add_argument("--scenario")
    r.add_argument("--text")
    r.add_argument("--seed", type=int, default=0)
    r.set_defaults(fn=cmd_run)

    t = sub.add_parser("train", help="PPO-train a residual policy")
    t.add_argument("--steps", type=int, default=200_000)
    t.add_argument("--out", default="runs/policy.npz")
    t.add_argument("--init")
    t.add_argument("--suite", help="directory of failure cases to replay")
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(fn=cmd_train)

    for name, fn in (("audit", cmd_audit), ("harden", cmd_harden)):
        s = sub.add_parser(name)
        s.add_argument("--policy")
        s.add_argument("--out", default=f"runs/{name}")
        s.add_argument("--title", default="Quadruped readiness")
        s.add_argument("--hunt-gens", type=int, default=5)
        s.add_argument("--workers", type=int, default=3)
        s.add_argument("--seed", type=int, default=0)
        if name == "audit":
            s.add_argument("--baseline", action="store_true", help="also audit the bare CPG gait")
        else:
            s.add_argument("--steps", type=int, default=120_000)
            s.add_argument("--profile", help="comma-separated deployment profiles to train toward, e.g. sidewalk,construction")
        s.set_defaults(fn=fn)

    g = sub.add_parser("regress", help="run a saved failure suite (CI gate)")
    g.add_argument("--policy")
    g.add_argument("--suite", required=True)
    g.set_defaults(fn=cmd_regress)

    lb = sub.add_parser("leaderboard", help="audit up to 4 policies side by side ('cpg' = baseline)")
    lb.add_argument("policies", nargs="+")
    lb.add_argument("--out", default="runs/leaderboard")
    lb.add_argument("--title", default="Policy leaderboard")
    lb.add_argument("--hunt-gens", type=int, default=5)
    lb.add_argument("--workers", type=int, default=3)
    lb.set_defaults(fn=cmd_leaderboard)

    ce = sub.add_parser("certify", help="audit + traceable readiness certificate (exit 1 below --min-score)")
    ce.add_argument("--policy")
    ce.add_argument("--out", default="runs/certificate.json")
    ce.add_argument("--min-score", type=float, default=70.0)
    ce.add_argument("--hunt-gens", type=int, default=5)
    ce.add_argument("--deploy-trials", type=int, default=DEPLOY_TRIALS)
    ce.add_argument("--workers", type=int, default=3)
    ce.set_defaults(fn=cmd_certify)

    dp = sub.add_parser("deploy", help="failure probability under a deployment profile (exit 1 if over budget)")
    dp.add_argument("--policy")
    dp.add_argument("--profile", default="all", choices=["all", *sorted(PROFILES)])
    dp.add_argument("--trials", type=int, default=60)
    dp.add_argument("--max-fail", type=float, default=0.10, help="allowed 95%% upper bound on failure rate")
    dp.add_argument("--seed", type=int, default=0)
    dp.add_argument("--workers", type=int, default=3)
    dp.add_argument("--out")
    dp.set_defaults(fn=cmd_deploy)

    qz = sub.add_parser("quantize", help="FP16/INT8 policy compression with a robustness check")
    qz.add_argument("--policy", required=True)
    qz.add_argument("--out", default="runs/quantize")
    qz.add_argument("--gated", action="store_true", help="audit behind the latency gate")
    qz.add_argument("--trials", type=int, default=30)
    qz.add_argument("--workers", type=int, default=3)
    qz.set_defaults(fn=cmd_quantize)

    fl = sub.add_parser("fleet", help="fit a site profile from shift telemetry (JSONL) and re-run humanoid odds")
    src = fl.add_mutually_exclusive_group(required=True)
    src.add_argument("--log", help="JSON Lines file, one record per shift")
    src.add_argument("--demo", help="generate a synthetic log from a built-in job profile")
    fl.add_argument("--shifts", type=int, default=300)
    fl.add_argument("--trials", type=int, default=150)
    fl.add_argument("--seed", type=int, default=0)
    fl.add_argument("--write-demo", help="also save the synthetic log here")
    fl.add_argument("--out")
    fl.set_defaults(fn=cmd_fleet)

    de = sub.add_parser("design", help="which spec change buys the most push robustness, job by job (paired)")
    de.add_argument("--trials", type=int, default=150)
    de.add_argument("--seed", type=int, default=0)
    de.add_argument("--out", default="runs/humanoid/levers.json")
    de.set_defaults(fn=cmd_design)

    hz = sub.add_parser("hazards", help="hazard log with measured frequencies from the humanoid lab")
    hz.add_argument("--trials", type=int, default=150)
    hz.add_argument("--seed", type=int, default=0)
    hz.add_argument("--out", default="runs/humanoid/hazards.md")
    hz.add_argument("--fail-on-high", action="store_true", help="exit 1 if any hazard rates high")
    hz.set_defaults(fn=cmd_hazards)

    hu = sub.add_parser("humanoid", help="humanoid push recovery: biped vs wheeled envelopes and job odds")
    hu.add_argument("--payload", type=float, default=0.0, help="tote held in front (kg)")
    hu.add_argument("--friction", type=float, default=0.8)
    hu.add_argument("--latency-ms", type=float, default=0.0)
    hu.add_argument("--trials", type=int, default=200)
    hu.add_argument("--seed", type=int, default=0)
    hu.add_argument("--out")
    hu.set_defaults(fn=cmd_humanoid)

    rt = sub.add_parser("redteam", help="compound-stress + red-team attacks on a policy")
    rt.add_argument("--policy")
    rt.add_argument("--n", type=int, default=5)
    rt.add_argument("--workers", type=int, default=3)
    rt.add_argument("--offline", action="store_true")
    rt.set_defaults(fn=cmd_redteam)

    v = sub.add_parser("serve", help="web playground")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8000)
    v.set_defaults(fn=cmd_serve)

    a = p.parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    raise SystemExit(main())
