#!/usr/bin/env bash
# Reproduces every committed result in runs/ (CPU only, ~40 min on 4 cores).
# Each step is independent; comment out what you don't need.
set -euo pipefail
cd "$(dirname "$0")/.."

# 1. v1 lineage. NOTE: the committed v1 was trained before the learner fixes (truncation
#    bootstrap, terrain variety, best-checkpoint selection), so a re-run gives a stronger v1.
python -m breakpoint train   --steps 240000 --out runs/policy_v1.npz --seed 0
python -m breakpoint harden  --policy runs/policy_v1.npz --out runs/hardened  --steps 100000 --hunt-gens 4
python -m breakpoint harden  --policy runs/hardened/policy_hardened.npz --out runs/hardened2 --steps 200000 --hunt-gens 5 --seed 7

# 2. v2: fixed learner + deterministic model selection, then deployment-aware hardening
python -m breakpoint train   --steps 420000 --out runs/v2/policy_v2.npz --seed 11
python -m breakpoint harden  --policy runs/v2/policy_v2.npz --out runs/v2field --steps 240000 --hunt-gens 4 \
                             --profile sidewalk,construction --seed 21

# 2b. latency-focused PPO round (did not beat the original; the selection guard keeps it)
python -m breakpoint harden  --policy runs/v2field/policy_v2field.npz --out runs/v3lat --steps 240000 --hunt-gens 4 \
                             --profile sidewalk,teleop --seed 33

# 3. side-by-side audits and deployment checks
python -m breakpoint leaderboard cpg runs/hardened/policy_hardened.npz runs/v2/policy_v2.npz \
                             runs/v2field/policy_v2field.npz --out runs/leaderboard2
python -m breakpoint leaderboard cpg runs/v2field/policy_v2field.npz gated:runs/v2field/policy_v2field.npz \
                             --out runs/leaderboard3
for p in cpg runs/hardened/policy_hardened.npz runs/v2/policy_v2.npz runs/v2field/policy_v2field.npz \
         gated:runs/v2field/policy_v2field.npz; do
  python -m breakpoint deploy --policy "$p" --trials 60 || true   # non-zero = over the failure budget
done

# 4. regression gate on the shrunk counterexamples
python -m breakpoint regress --policy runs/v2field/policy_v2field.npz --suite runs/v2field/suite || true
