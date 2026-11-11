# Hazard log (generated)

Frequencies are measured: 150 sampled shifts per job and body in the humanoid lab, with a 95% upper bound. Severity is assigned per hazard class and must be reviewed by a safety engineer. Hazards with no recorded events are not listed; that is evidence of rarity within the budget, not proof of absence.

| Risk | ID | Hazard | Job | Body | Events | Rate (≤ 95%) | Severity | Mitigation | Verify |
|---|---|---|---|---|---|---|---|---|---|
| high | H4 | Route blocked by a curb, threshold or stair | home care | wheeled | 70/150 | 46.7% (≤ 53.4%) | 2 | Route survey and ramps, or a legged platform where steps exist | `python -m breakpoint humanoid  # job table: trailer unloading, home care` |
| high | H4 | Route blocked by a curb, threshold or stair | trailer unloading | wheeled | 24/150 | 16.0% (≤ 21.5%) | 2 | Route survey and ramps, or a legged platform where steps exist | `python -m breakpoint humanoid  # job table: trailer unloading, home care` |
| medium | H1 | Fall after a shove the robot cannot step out of | teleoperated data collection | biped | 1/150 | 0.7% (≤ 2.9%) | 3 | Speed and separation monitoring near people; controlled-fall behaviour; widen step reach and swing speed | `python -m breakpoint humanoid  # push envelope per direction` |
