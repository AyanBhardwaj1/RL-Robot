# Breakpoint

**Find where your legged robot breaks before reality does.**

Breakpoint is a CPU-only robustness lab for learned legged-locomotion policies. You can describe a
real place in plain English, or upload a photo of the ground. Breakpoint compiles that into a
validated physics scenario and runs a quadruped through it in MuJoCo. It then:

1. **measures breaking points** on every stress axis,
2. **hunts for the mildest worlds that make the robot fail**,
3. **shrinks each failure** to a minimal counterexample and saves it as a regression test,
4. **retrains the policy** with PPO on those failures,
5. and writes a **readiness report** comparing the policy before and after retraining.

![playground](docs/playground.png)

## Why

A locomotion policy that looks fine in a demo usually fails somewhere specific: a side shove
during the wrong gait phase, a 60 ms Wi-Fi hiccup, a front-heavy backpack on wet tile. Breakpoint
turns "it looked fine" into numbers you can gate a release on. For example:

- "survives side shoves up to 5.6 N·s"
- "walks down to friction 0.22"
- "nominal failure rate ≤ 3.1% at 95% confidence"

It does this with no GPU and no PyTorch. The PPO learner is about 150 lines of numpy with
hand-written backprop, and that backprop is checked against finite differences in the tests.

## What's inside

| Module | What it does |
|---|---|
| `scenario.py` | One frozen, validated schema for every world. Every input is clamped to physical bounds: LLM output, HTTP bodies, files and search proposals. It also scores severity and finds the dominant stressor. |
| `world.py` | A procedural 12-DoF quadruped in MJCF, plus heightfield terrain (flat, rough, slope, stairs) and a payload. |
| `env.py` | A 50 Hz policy over 200 Hz physics. The policy outputs a residual on top of a trotting CPG, and PD torques are clipped by motor strength. Also handles action latency, observation noise and timed torso pushes. |
| `nn.py`, `ppo.py` | A numpy MLP, Adam optimizer and clipped-surrogate PPO with GAE and running observation normalisation. The scenario sampler mixes about 50% nominal randomisation, 30% replayed failures and 20% widened ranges. |
| `compiler.py` | Turns language or an image into a Scenario. A rule parser works offline. An optional OpenAI-compatible LLM (Llama API, vLLM, Ollama…) fills a bounded JSON schema and never writes simulator code. Any LLM error falls back to the rules. Photos go through explainable surface cues (texture, glare, colour, banding) or a vision LLM. |
| `probe.py` | Breaking-point bisection on 17 axes: 8 push directions, friction, payload, motor weakness, latency, sensor noise, uphill, downhill, stairs and roughness. Every push is tested at 4 gait phases, and a push margin is only as good as its worst phase. |
| `hunt.py` | Cross-entropy search for the score `margin + λ·severity`, so it finds *cheap* failures. A MAP-Elites atlas keeps the mildest failure per (stressor × severity band) cell. `shrink()` bisects each failure toward nominal, like property-based testing. |
| `redteam.py` | Compound-stress tests (pairs of the weakest axes at 50%/75% of their breaking points), a 2-D failure-boundary heatmap, and an LLM red team that proposes realistic multi-stressor situations from the measured weak spots. Offline, deterministic templates are used instead. |
| `report.py` | Readiness score (0–100), a grade (FIELD-READY / SUPERVISED / LAB-ONLY / NOT READY), a push-recovery polar plot, breaking-point bars, the failure atlas, and the search budget. Failure-rate bounds use the exact binomial (rule of three) or Wilson. Self-contained HTML. |
| `server.py`, `static/` | FastAPI and a three.js playground. You can compile a scenario, watch the replay, or use **Shove mode**, where you push the robot live over a WebSocket and score how much impulse it absorbs before it falls. A **ghost** mode replays a second policy as a translucent overlay in the same world. |

## Quick start

```bash
pip install -e '.[dev]'
python -m breakpoint compile "icy loading dock, carrying 3kg, shoved hard from the left at 3s"
python -m breakpoint run --text "stairs with 7 cm steps"              # exit code 1 if it fails
python -m breakpoint train --steps 240000 --out runs/policy_v1.npz     # ~5 min on 1 CPU core
python -m breakpoint audit --policy runs/policy_v1.npz --baseline     # report: runs/audit/report.html
python -m breakpoint harden --policy runs/policy_v1.npz --out runs/hardened
python -m breakpoint regress --policy runs/hardened/policy_hardened.npz --suite runs/hardened/suite  # CI gate
python -m breakpoint redteam --policy runs/hardened/policy_hardened.npz   # compound + red-team attacks
python -m breakpoint leaderboard cpg runs/policy_v1.npz runs/hardened/policy_hardened.npz
python -m breakpoint serve                                            # http://127.0.0.1:8000
pytest -q
```

To use an LLM for scenario compilation (optional):

```bash
export BREAKPOINT_LLM_BASE_URL=https://api.llama.com/compat/v1   # any OpenAI-compatible endpoint
export BREAKPOINT_LLM_API_KEY=...
export BREAKPOINT_LLM_MODEL=<model id>
```

## Results: leaderboard (CPU, committed in `runs/leaderboard/`)

All four policies were audited in one run with the same code, seeds and search budget
(`python -m breakpoint leaderboard cpg runs/policy_v1.npz runs/hardened/policy_hardened.npz runs/hardened2/policy_hardened2.npz`).

| Policy | Readiness | Grade | Margin index | Nominal pass | Uphill | Payload | Side push L/R (worst gait phase) | Compound fails | Red-team fails (of 5) |
|---|---|---|---|---|---|---|---|---|---|
| CPG gait alone | 66.8 | LAB-ONLY | 0.50 | 96% | 9.1° | 3.3 kg | 4.4 / 4.4 N·s | 2 | 3 |
| Residual PPO v1 (4 min training) | 50.9 | LAB-ONLY | 0.15 | 96% | 0° | 0 kg | 5.0 / 6.3 N·s | 0 | 4 |
| **v1 + hardening round 1** | **77.2** | **SUPERVISED** | **0.56** | **100%** | **16.3°** | **6.3 kg** | **6.3 / 8.8 N·s** | **1** | **2** |
| v1 + hardening rounds 1+2 | 76.6 | SUPERVISED | 0.55 | 100% | 13.4° | 5.3 kg | 6.3 / 6.9 N·s | 1 | 5 |

![report](docs/report.png)

What the lab found:

- **v1 looked fine in training but wasn't.** It walked under stochastic training noise but
  stalled when run deterministically. The audit scored it below the bare gait it was built on.
- **One hardening round fixed it.** Round 1 replayed 15 shrunk counterexamples (sensor noise,
  payload, terrain, latency, weak motors, low friction) for 100k PPO steps. Afterwards 10 of the
  15 pass on both seeds, and the policy beats the CPG baseline on every summary metric.
- **Round 2 overfit.** It trained 200k more steps on 11 new counterexamples but fixed only 2 of
  them. Its uphill and payload margins shrank, and red-team failures rose from 2 to 5. Its own
  before/after report showed a small gain (77.3 → 77.8); the leaderboard's fresh adversarial
  search did not. **Ship round 1.**
- **Compound fragility is real.** For the hardened policy, *stairs + payload*, each at 75% of its
  own breaking point, fails on every seed, while each passes alone.
- **The failure-boundary map shows a counter-intuitive pocket.** At friction 0.05 the robot survives
  a 16 N·s side shove that topples it on concrete, because it slides instead of tripping. On that ice
  it also can't make forward progress at all, so it still fails the cell with no push.

![compound stress](docs/report_compound.png)

## The hardening loop

```
policy ──► probe (17 axes) ──► hunt (CEM + MAP-Elites atlas) ──► shrink ──► suite/*.json
   ▲                                                                           │
   └──────── PPO fine-tune (30% failure replay) ◄───────────────────────────────┘
                     │
                     └──► re-audit ──► report.html (before vs after) + regress (CI)
```

Each shrunk counterexample is a JSON file with a scenario and seed. `breakpoint regress` replays
the suite on 3 seeds and exits non-zero if any case fails, so a robot's failure history becomes
a unit-test suite.

## Honest limitations

- The robot is a generic 6 kg quadruped, not a calibrated model of any commercial platform. Its
  margins show *relative* robustness, not certified hardware limits.
- CPU training budgets are small (minutes, not billions of steps). A briefly trained residual
  policy can be **worse** than the bare CPG on some axes. The report is designed to catch that, and
  it does.
- Static search cannot prove the absence of failures. The report always states the search budget.
- The image heuristic is a cheap, explainable prior, not 3D reconstruction. A vision LLM improves it.

See `reports/Legged robot robustness platform.md` for the research behind these design choices.
