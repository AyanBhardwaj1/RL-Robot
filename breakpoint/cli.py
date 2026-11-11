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

import numpy as np
from pathlib import Path

from .compiler import compile_image, compile_text
from .evaluate import evaluate_many, rollout
from .hunt import hunt, shrink
from .policy import ResidualPolicy, load_policy
from .ppo import EVAL_WORLDS, PPOConfig, ScenarioSampler, deterministic_score, train
from .probe import probe
from .profiles import PROFILES, deployment_check
from .redteam import boundary_map, interactions, red_team
from .report import render_html, scorecard
from .scenario import Scenario

NOMINAL_TRIALS = 24


def _log(row: dict) -> None:
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)


def nominal_trials(spec: str | None, n: int = NOMINAL_TRIALS, seed: int = 123) -> list[dict]:
    sampler = ScenarioSampler(seed=seed, widen_frac=0.0)
    scns = [sampler.sample(6.0) for _ in range(n)]
    return [r[0] for r in evaluate_many(spec, scns, seeds=[seed])]


def audit(spec: str | None, label: str, hunt_gens: int = 5, workers: int = 3) -> dict:
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
    card["compound_failures"] = sum(c["compound_failure"] for c in extras["interactions"])
    card["redteam_failures"] = sum(a["pass_rate"] < 1 for a in extras["redteam"]["attacks"])
    return {"label": label, "policy": spec or "cpg", "probe": pr, "nominal": nom, "hunt": h.to_dict(),
            "card": card, "extras": extras}


def _weakest_push(pr: list[dict]) -> str:
    pushes = [r for r in pr if r["key"].startswith("push_")]
    return min(pushes, key=lambda r: r["fraction"])["key"] if pushes else "push_left"


def _write_report(out: Path, title: str, audits: list[dict], notes: list[str]) -> Path:
    last = audits[-1]
    page = render_html(title, {a["label"]: a["card"] for a in audits}, {a["label"]: a["probe"] for a in audits},
                       last["hunt"]["atlas"], last["hunt"], notes, last.get("extras"))
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


def cmd_leaderboard(a: argparse.Namespace) -> int:
    specs = [None if p == "cpg" else p for p in a.policies][:4]
    labels: list[str] = []
    for sp in specs:
        lab = "cpg-baseline" if sp is None else Path(sp).stem
        labels.append(lab if lab not in labels else f"{Path(sp).parent.name}/{lab}")
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
    before = audit(a.policy, "before", a.hunt_gens, a.workers)
    suite = save_suite(before["hunt"]["atlas"], a.policy, out / "suite")
    print(f"[harden] {len(suite)} minimal counterexamples saved to {out / 'suite'}", flush=True)
    failures = [Scenario.from_dict(json.loads(p.read_text())["scenario"]) for p in suite]
    pol = ResidualPolicy.load(a.policy) if a.policy else ResidualPolicy(seed=a.seed)
    pol.name = "hardened"
    hardened = out / f"policy_{out.name}.npz"
    profiles = [PROFILES[n] for n in (a.profile.split(",") if a.profile else [])]
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
    notes = [f"Hardening replayed {len(failures)} shrunk counterexamples (30% of episodes) for {a.steps} PPO steps; "
             f"{fixed}/{len(failures)} now pass on both seeds."]
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

    dp = sub.add_parser("deploy", help="failure probability under a deployment profile (exit 1 if over budget)")
    dp.add_argument("--policy")
    dp.add_argument("--profile", default="all", choices=["all", *sorted(PROFILES)])
    dp.add_argument("--trials", type=int, default=60)
    dp.add_argument("--max-fail", type=float, default=0.10, help="allowed 95%% upper bound on failure rate")
    dp.add_argument("--seed", type=int, default=0)
    dp.add_argument("--workers", type=int, default=3)
    dp.add_argument("--out")
    dp.set_defaults(fn=cmd_deploy)

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
