# Hazard log (generated)

Frequencies are measured: 150 sampled shifts per job and body in the humanoid lab, with a 95% upper bound. Severity is assigned per hazard class and must be reviewed by a safety engineer. Hazards with no recorded events are not listed; that is evidence of rarity within the budget, not proof of absence.

| Risk | ID | Hazard | Job | Body | Events | Rate (≤ 95%) | Severity | Mitigation | Verify |
|---|---|---|---|---|---|---|---|---|---|
| high | H4 | Route blocked by a curb, threshold or stair | home care | wheeled | 70/150 | 46.7% (≤ 53.4%) | 2 | Route survey and ramps, or a legged platform where steps exist | `python -m breakpoint humanoid  # job table: trailer unloading, home care` |
| high | H4 | Route blocked by a curb, threshold or stair | trailer unloading | wheeled | 24/150 | 16.0% (≤ 21.5%) | 2 | Route survey and ramps, or a legged platform where steps exist | `python -m breakpoint humanoid  # job table: trailer unloading, home care` |
| high | H2 | Loss of footing on a slippery floor | trailer unloading | biped | 16/150 | 10.7% (≤ 15.5%) | 3 | Floor-condition detection and a friction-aware gait; spill response procedure | `python -m breakpoint humanoid --friction 0.3` |
| high | H6 | Capture point runs away beyond leg reach | teleoperated data collection | biped | 6/150 | 4.0% (≤ 7.5%) | 3 | Same as H1; detect early and trigger a protective stop | `python -m breakpoint humanoid` |
| high | H2 | Loss of footing on a slippery floor | teleoperated data collection | biped | 6/150 | 4.0% (≤ 7.5%) | 3 | Floor-condition detection and a friction-aware gait; spill response procedure | `python -m breakpoint humanoid --friction 0.3` |
| high | H6 | Capture point runs away beyond leg reach | tote handling | biped | 5/150 | 3.3% (≤ 6.7%) | 3 | Same as H1; detect early and trigger a protective stop | `python -m breakpoint humanoid` |
| high | H1 | Fall after a shove the robot cannot step out of | teleoperated data collection | biped | 4/150 | 2.7% (≤ 5.8%) | 3 | Speed and separation monitoring near people; controlled-fall behaviour; widen step reach and swing speed | `python -m breakpoint humanoid  # push envelope per direction` |
| medium | H2 | Loss of footing on a slippery floor | line-side kitting | biped | 3/150 | 2.0% (≤ 4.9%) | 3 | Floor-condition detection and a friction-aware gait; spill response procedure | `python -m breakpoint humanoid --friction 0.3` |
| medium | H6 | Capture point runs away beyond leg reach | line-side kitting | biped | 2/150 | 1.3% (≤ 4.0%) | 3 | Same as H1; detect early and trigger a protective stop | `python -m breakpoint humanoid` |
| medium | H1 | Fall after a shove the robot cannot step out of | line-side kitting | biped | 2/150 | 1.3% (≤ 4.0%) | 3 | Speed and separation monitoring near people; controlled-fall behaviour; widen step reach and swing speed | `python -m breakpoint humanoid  # push envelope per direction` |
| medium | H2 | Loss of footing on a slippery floor | tote handling | biped | 2/150 | 1.3% (≤ 4.0%) | 3 | Floor-condition detection and a friction-aware gait; spill response procedure | `python -m breakpoint humanoid --friction 0.3` |
| medium | H1 | Fall after a shove the robot cannot step out of | trailer unloading | biped | 2/150 | 1.3% (≤ 4.0%) | 3 | Speed and separation monitoring near people; controlled-fall behaviour; widen step reach and swing speed | `python -m breakpoint humanoid  # push envelope per direction` |
| medium | H6 | Capture point runs away beyond leg reach | trailer unloading | biped | 2/150 | 1.3% (≤ 4.0%) | 3 | Same as H1; detect early and trigger a protective stop | `python -m breakpoint humanoid` |
| medium | H1 | Fall after a shove the robot cannot step out of | tote handling | biped | 1/150 | 0.7% (≤ 2.9%) | 3 | Speed and separation monitoring near people; controlled-fall behaviour; widen step reach and swing speed | `python -m breakpoint humanoid  # push envelope per direction` |
